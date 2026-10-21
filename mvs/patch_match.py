"""PatchMatch stereo (geom consistency), VRAM-adaptive, OOM-safe.

Runs pycolmap.patch_match_stereo in a *python subprocess*: the pip pycolmap
wheel is the only build whose CUDA initializes on the cluster's driver-545
nodes (the conda-forge CLI's CUDA 12.9 runtime reports "CUDA not available"
there), and the subprocess boundary keeps any native crash/OOM catchable as
a return code. On failure of the whole-scene run, retries in ref-image
chunks by rewriting stereo/patch-match.cfg (each ref image is processed
independently with `__auto__` source selection from the full model, so a
plain partition is equivalent to the monolithic run).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mvs_config as cfg

_STEREO_SCRIPT = """
import sys
import pycolmap
dense_dir, max_image_size = sys.argv[1], int(sys.argv[2])
opts = pycolmap.PatchMatchOptions()
opts.geom_consistency = {geom}
opts.cache_size = {cache}
if max_image_size > 0:
    opts.max_image_size = max_image_size
pycolmap.patch_match_stereo(workspace_path=dense_dir, options=opts)
"""


def _vram_gb() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip().split("\n")[0]) / 1024.0
    except Exception:  # noqa: BLE001
        return 0.0


def _stereo_cmd(dense_dir: Path, max_image_size: int) -> list[str]:
    script = _STEREO_SCRIPT.format(
        geom=cfg.GEOM_CONSISTENCY, cache=cfg.CACHE_SIZE_GB)
    return [sys.executable, "-c", script, str(dense_dir), str(max_image_size)]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", timeout=cfg.STEREO_TIMEOUT_S)


def _load_rate() -> float:
    try:
        return json.load(open(cfg.RATE_JSON))["sec_per_image"]
    except Exception:  # noqa: BLE001
        return cfg.DEFAULT_SEC_PER_IMAGE


def _update_rate(n_images: int, elapsed: float) -> None:
    try:
        prev = _load_rate()
        new = 0.7 * prev + 0.3 * (elapsed / max(n_images, 1))
        cfg.RATE_JSON.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"sec_per_image": round(new, 2)}, open(cfg.RATE_JSON, "w"))
    except Exception:  # noqa: BLE001
        pass


def eta_seconds(n_images: int) -> float:
    return n_images * _load_rate()


def _chunked_retry(dense_dir: Path, max_image_size: int) -> dict:
    """Rewrite patch-match.cfg per chunk of ref images and run sequentially."""
    cfg_path = dense_dir / "stereo" / "patch-match.cfg"
    original = cfg_path.read_text()
    shutil.copy(cfg_path, cfg_path.with_suffix(".cfg.orig"))
    lines = original.strip().split("\n")
    entries = [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]

    chunks = [entries[i:i + cfg.CHUNK_SIZE]
              for i in range(0, len(entries), cfg.CHUNK_SIZE)]
    try:
        for ci, chunk in enumerate(chunks):
            cfg_path.write_text(
                "\n".join(f"{a}\n{b}" for a, b in chunk) + "\n")
            proc = _run(_stereo_cmd(dense_dir, max_image_size))
            if proc.returncode != 0:
                raise RuntimeError(
                    f"chunk {ci + 1}/{len(chunks)} failed "
                    f"(rc={proc.returncode}):\n{proc.stderr[-1500:]}")
        return {"mode": "chunked", "n_chunks": len(chunks)}
    finally:
        cfg_path.write_text(original)


def run_stereo(dense_dir: Path, n_images: int) -> dict:
    vram = _vram_gb()
    max_image_size = (cfg.LOW_VRAM_MAX_IMAGE_SIZE
                      if 0 < vram < cfg.LOW_VRAM_GB else -1)
    print(f"[stereo] stereo: {n_images} imgs, VRAM {vram:.0f}G, "
          f"max_image_size={max_image_size}, "
          f"ETA ~{eta_seconds(n_images) / 60:.0f} min", flush=True)

    t0 = time.time()
    proc = _run(_stereo_cmd(dense_dir, max_image_size))
    if proc.returncode == 0:
        info = {"mode": "whole_scene"}
    else:
        print(f"[stereo] whole-scene stereo failed (rc={proc.returncode}) — "
              f"retrying in chunks of {cfg.CHUNK_SIZE}", flush=True)
        info = _chunked_retry(dense_dir, max_image_size)

    elapsed = time.time() - t0
    _update_rate(n_images, elapsed)
    n_depth = len(list((dense_dir / "stereo" / "depth_maps")
                       .glob("*.geometric.bin")))
    info.update(vram_gb=round(vram, 1), max_image_size=max_image_size,
                stereo_s=round(elapsed, 1), n_geometric_bins=n_depth)
    return info
