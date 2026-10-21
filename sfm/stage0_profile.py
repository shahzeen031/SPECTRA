"""Stage 0 — per-scene profiling (CPU only).

Computes: frame count, image resolution (asserts single resolution per scene),
texture proxy (mean Laplacian variance over sampled frames), TOO_SMALL flag.
Writes {workspace}/profile.json.
"""
import json

import cv2
import numpy as np
from PIL import Image

import config
from scene_index import Scene, list_frames


def _laplacian_variance(img_path: str) -> float:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape
    if w > config.TEXTURE_DOWNSCALE_W:
        scale = config.TEXTURE_DOWNSCALE_W / w
        img = cv2.resize(img, (config.TEXTURE_DOWNSCALE_W, int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def profile_scene(scene: Scene, force: bool = False) -> dict:
    out_path = scene.workspace / "profile.json"
    if out_path.exists() and not force:
        with open(out_path) as f:
            return json.load(f)

    frames = list_frames(scene.image_dir)
    n = len(frames)

    # resolution check: read image *headers* only (fast, no decode)
    sizes = set()
    for name in frames:
        with Image.open(scene.image_dir / name) as im:
            sizes.add(im.size)
    if len(sizes) != 1:
        raise RuntimeError(
            f"{scene.name}: mixed resolutions {sizes} — pipeline assumes one per scene")
    width, height = sizes.pop()

    # texture proxy over uniformly sampled frames
    idx = np.linspace(0, n - 1, min(config.TEXTURE_SAMPLE_FRAMES, n)).astype(int)
    texture = float(np.mean([
        _laplacian_variance(str(scene.image_dir / frames[i])) for i in idx]))

    fx, fy, cx, cy = config.read_intrinsics(scene.image_dir.parent)
    if not (0 < cx < width and 0 < cy < height):
        raise RuntimeError(
            f"{scene.name}: principal point ({cx}, {cy}) outside the "
            f"{width}x{height} images — intrinsics must match the rgb/ resolution")
    profile = {
        "scene": scene.name,
        "weather": scene.weather,
        "env": scene.env,
        "n_frames": n,
        "width": width,
        "height": height,
        "camera_params": [fx, fy, cx, cy],
        "texture": texture,
        "too_small": n < config.MIN_FRAMES,
    }
    scene.workspace.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(profile, fh, indent=2)
    return profile


def compute_texture_quartiles(profiles: list[dict]) -> dict:
    """Global texture stats across all scenes; used for auto-aggressive rule."""
    vals = sorted(p["texture"] for p in profiles)
    q25 = float(np.percentile(vals, 25)) if vals else 0.0
    stats = {"q25": q25, "n_scenes": len(vals)}
    with open(config.TEXTURE_STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_texture_quartiles() -> dict:
    if not config.TEXTURE_STATS_JSON.exists():
        raise RuntimeError(
            f"{config.TEXTURE_STATS_JSON} missing — run '--stage profile' first "
            "(it must cover ALL scenes so the aggressive-tier quartile rule is global)")
    with open(config.TEXTURE_STATS_JSON) as f:
        return json.load(f)


def wants_auto_aggressive(profile: dict, texture_q25: float) -> bool:
    """Route straight to the aggressive feature tier (skip plain SIFT)."""
    return (profile["weather"] in config.AUTO_AGGRESSIVE_WEATHERS
            and profile["texture"] <= texture_q25)
