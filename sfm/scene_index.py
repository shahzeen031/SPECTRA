"""Scene discovery.

Layout: {DATA_ROOT}/{scene}/rgb/*.jpg|png   (+ {DATA_ROOT}/{scene}/x/ for the
second modality, used only when building pairs). Scene name = folder name.
Weather / environment are optional hints parsed from the name
(e.g. 1_sunny_data_1_urban); "unknown" otherwise.
"""
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import config

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
WEATHERS = ("sunny", "cloudy", "rainy", "snowy")


@dataclass
class Scene:
    name: str          # e.g. 1_sunny_data_1_urban
    weather: str       # sunny | cloudy | rainy | snowy
    env: str           # urban | suburban | unknown
    image_dir: Path    # {DATA_ROOT}/{scene}/rgb
    n_frames: int

    @property
    def workspace(self) -> Path:
        return config.OUTPUT_ROOT / self.name


def list_frames(image_dir: Path) -> list[str]:
    """Sorted image filenames only (excludes Thumbs.db and other junk)."""
    return sorted(
        f.name for f in image_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS and f.is_file()
    )


def _parse_weather(name: str) -> str:
    for w in WEATHERS:
        if re.search(rf"(^|_){w}(_|$)", name.lower()):
            return w
    return "unknown"


def _parse_env(scene_dir: str) -> str:
    for env in ("suburban", "urban"):
        if scene_dir.endswith(env):
            return env
    return "unknown"


def discover_scenes(root: Path = None) -> list[Scene]:
    root = root or config.DATA_ROOT
    scenes = []
    for rgb_dir in sorted(root.glob("*/rgb")):
        name = rgb_dir.parent.name
        frames = list_frames(rgb_dir)
        if not frames:
            continue
        scenes.append(Scene(
            name=name,
            weather=_parse_weather(name),
            env=_parse_env(name),
            image_dir=rgb_dir,
            n_frames=len(frames),
        ))
    return scenes


def write_inventory(scenes: list[Scene], path: Path = None) -> None:
    path = path or config.INVENTORY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "weather", "env", "n_frames", "image_dir"])
        for s in scenes:
            w.writerow([s.name, s.weather, s.env, s.n_frames, str(s.image_dir)])


def filter_scenes(scenes: list[Scene], names: list[str] | None = None,
                  weathers: list[str] | None = None) -> list[Scene]:
    out = scenes
    if names:
        wanted = set(names)
        out = [s for s in out if s.name in wanted]
    if weathers:
        wanted = {w.lower() for w in weathers}
        out = [s for s in out if s.weather in wanted]
    return out
