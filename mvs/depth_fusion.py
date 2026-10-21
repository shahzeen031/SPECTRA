"""Depth fusion — geo/photo fusion, stability filter, evidence-based semantic
masking. Adapted from the MegaDepth post-processing recipe
(demo_code/semantic_filtering.m) for driving scenes:

  1. geo/photo "closer pixel" fusion (MegaDepth: replace geo with photo
     wherever geo/photo > 1.1 — recovers coverage lost to pure geometric
     filtering).
  2. Stability filter: double 7x7 nan-median filter (GPU), reject/replace
     pixels whose filtered value deviates from the original by > 1.1x
     (symmetric ratio) — matches MegaDepth's spike rejection.
  3. Semantic masking, evidence-based rather than blanket-by-class:
       - sky: always masked
       - road/sidewalk/terrain: kept where MVS-valid AND within the
         per-scene geometric reliability range; masked (not discarded)
         beyond that range
       - person/rider/car/truck/bus/train/motorcycle/bicycle: per
         connected component, kept iff the component's already-valid
         MVS fraction >= COMPONENT_EVIDENCE_MIN_VALID (static/parked
         survives; genuinely moving objects fail MVS geometric
         consistency and get dropped)
       - building/wall/fence/pole/traffic light/sign/vegetation: kept
         untouched

Per-pixel `source` channel: 2 = MVS (post geo/photo fusion + stability +
semantic masking), 0 = invalid/masked.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label

import fusion_config as cfg


# ------------------------------------------------------------------ per-scene geometry
def ground_range_cutoff(rec, focal_px: float) -> float:
    """baseline*focal/min_disparity, using median inter-frame camera-center
    motion as a proxy for the baseline PatchMatch's auto neighbor selection
    would typically use."""
    ids = sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name)
    centers = []
    for iid in ids:
        im = rec.images[iid]
        p = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = np.asarray(p.rotation.matrix() if callable(p.rotation.matrix)
                       else p.rotation.matrix)
        t = np.asarray(p.translation)
        centers.append(-R.T @ t)
    centers = np.asarray(centers)
    if len(centers) < 2:
        return float("inf")
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    baseline = float(np.median(steps[steps > 0])) if (steps > 0).any() else 0.0
    if baseline <= 0:
        return float("inf")
    return cfg.GROUND_RANGE_SAFETY * baseline * focal_px / cfg.MIN_DISPARITY_PX


# ------------------------------------------------------------------ GPU median filter
def _nan_median_filter_gpu(depth: np.ndarray, size: int, device: str) -> np.ndarray:
    x = torch.from_numpy(depth).to(device).float()
    x = torch.where(x > 0, x, torch.tensor(float("nan"), device=device))
    pad = size // 2
    xp = F.pad(x[None, None], (pad, pad, pad, pad), mode="reflect")
    patches = xp.unfold(2, size, 1).unfold(3, size, 1)
    patches = patches.reshape(1, 1, patches.shape[2], patches.shape[3], size * size)
    med = torch.nanmedian(patches, dim=-1).values[0, 0]
    med = torch.nan_to_num(med, nan=0.0)
    return med.cpu().numpy()


# ------------------------------------------------------------------ robust affine fit
def fuse_frame(geo: np.ndarray, photo: np.ndarray, seg: np.ndarray,
              ground_cutoff: float, device: str
              ) -> tuple[np.ndarray, np.ndarray, dict]:
    assert geo.shape == photo.shape == seg.shape, "shape mismatch geo/photo/seg"
    stats = {}

    # 1. geo/photo closer-pixel fusion
    fused = geo.copy()
    both = (geo > 0) & (photo > 0)
    ratio = np.zeros_like(geo)
    ratio[both] = geo[both] / np.maximum(photo[both], 1e-9)
    replace = both & (ratio > cfg.GEO_PHOTO_RATIO)
    fused[replace] = photo[replace]
    stats["n_photo_rescued"] = int(replace.sum())

    # 2. stability filter — double median filter, reject large deviations
    work = fused
    for _ in range(cfg.MEDIAN_FILTER_PASSES):
        work = _nan_median_filter_gpu(work, cfg.MEDIAN_FILTER_SIZE, device)
    valid0 = fused > 0
    dev_ratio = np.ones_like(fused)
    nz = valid0 & (work > 0)
    r = np.zeros_like(fused)
    r[nz] = work[nz] / fused[nz]
    dev_ratio[nz] = np.maximum(r[nz], 1.0 / r[nz])
    unstable = valid0 & (~nz | (dev_ratio > cfg.STABILITY_RATIO))
    fused = work.copy()
    fused[unstable] = 0.0
    stats["n_stability_rejected"] = int(unstable.sum())

    # 3. semantic masking
    sky = np.isin(seg, list(cfg.SKY_CLASSES))
    n_before_ground = int((fused > 0).sum())
    ground = np.isin(seg, list(cfg.GROUND_CLASSES))
    too_far = ground & (fused > ground_cutoff)
    fused[too_far] = 0.0
    stats["n_ground_range_masked"] = int(too_far.sum())

    dyn = np.isin(seg, list(cfg.DYNAMIC_CLASSES))
    labeled, n_comp = cc_label(dyn)
    n_dyn_kept, n_dyn_dropped = 0, 0
    for comp_id in range(1, n_comp + 1):
        comp = labeled == comp_id
        n_px = int(comp.sum())
        if n_px < cfg.COMPONENT_MIN_SIZE:
            fused[comp] = 0.0
            n_dyn_dropped += n_px
            continue
        valid_frac = float((fused[comp] > 0).mean())
        if valid_frac < cfg.COMPONENT_EVIDENCE_MIN_VALID:
            fused[comp] = 0.0
            n_dyn_dropped += n_px
        else:
            n_dyn_kept += n_px
    stats["n_dynamic_components"] = int(n_comp)
    stats["n_dynamic_kept_px"] = n_dyn_kept
    stats["n_dynamic_dropped_px"] = n_dyn_dropped
    fused[sky] = 0.0

    source = np.where(fused > 0, 2, 0).astype(np.uint8)

    # 4. non-sky valid fraction (MegaDepth's weak-flag metric)
    non_sky = ~sky
    n_non_sky = int(non_sky.sum())
    valid_non_sky = float((fused[non_sky] > 0).sum() / max(n_non_sky, 1))
    weak = valid_non_sky < cfg.NON_SKY_VALID_FRACTION_WEAK
    stats["valid_fraction_non_sky"] = valid_non_sky
    stats["weak"] = bool(weak)

    stats["n_final_valid"] = int((fused > 0).sum())
    stats["n_before_semantic_masking"] = n_before_ground
    return fused.astype(np.float32), source, stats
