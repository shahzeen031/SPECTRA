"""Undistortion — model verification + undistortion into {workspace}/dense/.

Chooses sparse/full (PnP-densified) when present and loadable, else falls
back to sparse/best with mvs_model=best_fallback. Verifies the locked
PINHOLE intrinsics before any GPU time is spent.
"""
import json
from pathlib import Path

import numpy as np
import pycolmap

import mvs_config as cfg
from mvs_scenes import MvsScene


def _try_load(model_dir: Path) -> pycolmap.Reconstruction | None:
    if not model_dir.exists():
        return None
    try:
        rec = pycolmap.Reconstruction(model_dir)
        return rec if rec.num_reg_images() > 0 else None
    except Exception:  # noqa: BLE001 — corrupt model -> fallback
        return None


def verify_model(rec: pycolmap.Reconstruction, expected: list[float]) -> dict:
    """Assert PINHOLE + locked-intrinsics drift < 1e-6."""
    cams = list(rec.cameras.values())
    if len(cams) != 1:
        raise RuntimeError(f"expected 1 shared camera, found {len(cams)}")
    cam = cams[0]
    if "PINHOLE" not in str(cam.model):
        raise RuntimeError(f"camera model is {cam.model}, expected PINHOLE")
    params = np.asarray(cam.params, dtype=float)   # fx fy cx cy
    exp = np.asarray(expected, dtype=float)
    drift = float(np.abs(params - exp).max())
    if drift > cfg.FOCAL_DRIFT_TOL:
        raise RuntimeError(
            f"intrinsics drift {drift:.3e} > {cfg.FOCAL_DRIFT_TOL} "
            f"(model {params.tolist()} vs locked {exp.tolist()})")
    return {"width": cam.width, "height": cam.height,
            "params": params.tolist(), "drift": drift}


def run_undistort(scene: MvsScene) -> dict:
    """Returns {model_used, model_dir, n_images, camera, dense_dir}."""
    model_used, model_dir, rec = "full", scene.sfm_dir / "sparse" / "full", None
    rec = _try_load(model_dir)
    if rec is None:
        model_used = "best_fallback"
        model_dir = scene.sfm_dir / "sparse" / "best"
        rec = _try_load(model_dir)
        if rec is None:
            raise RuntimeError("neither sparse/full nor sparse/best loadable")

    with open(scene.sfm_dir / "profile.json") as f:
        expected = json.load(f)["camera_params"]
    cam_info = verify_model(rec, expected)
    n_images = rec.num_reg_images()

    # Reuse a COMPLETE undistortion (image count matches + stereo config
    # present): patch-match natively skips already-computed views, so a
    # killed job's stereo progress survives a resume. Anything partial or
    # inconsistent is wiped — a half-written dense dir crashes the
    # undistorter natively.
    import shutil
    dense_dir = scene.workspace / "dense"
    img_dir, pm_cfg = dense_dir / "images", dense_dir / "stereo" / "patch-match.cfg"
    if img_dir.exists() and pm_cfg.exists():
        n_existing = sum(1 for p in img_dir.iterdir() if p.is_file())
        if n_existing == n_images:
            return {"model_used": model_used, "model_dir": str(model_dir),
                    "n_images": n_images, "camera": cam_info,
                    "dense_dir": str(dense_dir), "reused": True}
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True)

    # Subprocess for crash isolation (same rationale as patch_match.py): a native abort
    # becomes a catchable return code -> scene-level MVS_FAIL, not a dead job.
    import subprocess, sys
    script = (
        "import sys, pycolmap\n"
        "pycolmap.undistort_images(output_path=sys.argv[1], "
        "input_path=sys.argv[2], image_path=sys.argv[3], "
        f"num_threads={cfg.NUM_THREADS})\n")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(dense_dir), str(model_dir),
         str(scene.image_dir)],
        capture_output=True, text=True, errors="replace", timeout=2 * 3600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"undistort subprocess failed (rc={proc.returncode}):\n"
            f"{proc.stderr[-1500:]}")

    # PINHOLE + zero distortion => copy/crop; count must match the model
    n_out = sum(1 for p in (dense_dir / "images").iterdir() if p.is_file())
    if n_out != n_images:
        raise RuntimeError(
            f"undistorted image count {n_out} != model images {n_images}")

    return {"model_used": model_used, "model_dir": str(model_dir),
            "n_images": n_images, "camera": cam_info,
            "dense_dir": str(dense_dir)}
