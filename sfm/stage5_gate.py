"""Stage 5 — density-normalized, tiered quality gate.

Metrics: reg_rate, points-per-registered-image, mean reprojection error
(sampled up to 200 observations from EACH registered image), fx/fy drift.

PASS: reg_rate >= 0.60 AND pts/img >= 40 AND reproj < 0.75
WARN: same but reproj in [0.75, 1.10]
FAIL: otherwise
"""
import random
from pathlib import Path

import numpy as np
import pycolmap

import config


def _image_reproj_error(rec: pycolmap.Reconstruction, image,
                        max_obs: int = 200) -> list[float]:
    """Reprojection errors for up to max_obs sampled observations of one image."""
    cam = rec.cameras[image.camera_id]
    obs = [p for p in image.points2D if p.has_point3D()]
    if not obs:
        return []
    if len(obs) > max_obs:
        obs = random.sample(obs, max_obs)
    try:
        pose = image.cam_from_world() if callable(image.cam_from_world) \
            else image.cam_from_world
        errs = []
        for p2d in obs:
            xyz = rec.points3D[p2d.point3D_id].xyz
            xyz_cam = pose * xyz
            if xyz_cam[2] <= 0:
                continue
            uv = None
            try:                       # newer binding: 3D point in cam frame
                uv = cam.img_from_cam(xyz_cam)
            except (TypeError, RuntimeError):
                try:                   # older binding: normalized 2D
                    uv = cam.img_from_cam(xyz_cam[:2] / xyz_cam[2])
                except (TypeError, RuntimeError):
                    pass
            if uv is None:
                raise RuntimeError("img_from_cam API mismatch")
            errs.append(float(np.linalg.norm(np.asarray(uv).ravel()[:2] - p2d.xy)))
        return errs
    except Exception:  # noqa: BLE001 — fall back to track errors
        return [float(rec.points3D[p.point3D_id].error) for p in obs
                if rec.points3D[p.point3D_id].error >= 0]


def evaluate(best_dir: Path, n_selected: int,
             expected_params: list[float]) -> dict:
    rec = pycolmap.Reconstruction(best_dir)
    reg_ids = list(rec.reg_image_ids())
    n_reg = len(reg_ids)
    reg_rate = n_reg / max(n_selected, 1)

    pts_counts, errs = [], []
    for img_id in reg_ids:
        image = rec.images[img_id]
        n3d = sum(1 for p in image.points2D if p.has_point3D())
        pts_counts.append(n3d)
        errs.extend(_image_reproj_error(rec, image))
    pts_per_img = float(np.mean(pts_counts)) if pts_counts else 0.0
    reproj = float(np.mean(errs)) if errs else float("inf")

    # fx/fy drift vs configured fixed intrinsics
    drift = 0.0
    for cam in rec.cameras.values():
        params = np.asarray(cam.params, dtype=float)
        exp = np.asarray(expected_params, dtype=float)
        if params.shape == exp.shape:
            drift = max(drift, float(np.abs(params[:2] - exp[:2]).max()))
    intrinsics_fixed = drift < config.FOCAL_DRIFT_TOL

    density_ok = (reg_rate >= config.GATE_MIN_REG_RATE
                  and pts_per_img >= config.GATE_MIN_PTS_PER_IMG)
    if density_ok and reproj < config.GATE_PASS_REPROJ:
        tier = "PASS"
    elif density_ok and reproj <= config.GATE_WARN_REPROJ:
        tier = "WARN"
    else:
        tier = "FAIL"

    return {
        "tier": tier,
        "n_registered": n_reg,
        "reg_rate": round(reg_rate, 4),
        "pts_per_img": round(pts_per_img, 1),
        "reproj_error": round(reproj, 4) if np.isfinite(reproj) else None,
        "num_points3D": rec.num_points3D(),
        "focal_drift": drift,
        "intrinsics_fixed": intrinsics_fixed,
        "fail_reasons": _fail_reasons(reg_rate, pts_per_img, reproj),
    }


def _fail_reasons(reg_rate, pts_per_img, reproj) -> list[str]:
    reasons = []
    if reg_rate < config.GATE_MIN_REG_RATE:
        reasons.append(f"reg_rate {reg_rate:.2f} < {config.GATE_MIN_REG_RATE}")
    if pts_per_img < config.GATE_MIN_PTS_PER_IMG:
        reasons.append(f"pts/img {pts_per_img:.0f} < {config.GATE_MIN_PTS_PER_IMG}")
    if not (reproj <= config.GATE_WARN_REPROJ):
        reasons.append(f"reproj {reproj:.2f} > {config.GATE_WARN_REPROJ}")
    return reasons
