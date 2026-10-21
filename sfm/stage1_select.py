"""Stage 1 — frame selection.

N <= 400            : all frames
400 < N <= 1200     : uniform stride = ceil(N/400), full trajectory span
N > 1200            : motion-adaptive to a 400-frame budget (Farneback flow)

Persists {workspace}/selected_frames.txt and returns (names, meta).
"""
import math

import cv2
import numpy as np

import config
from scene_index import Scene, list_frames


def _flow_magnitudes(scene: Scene, frames: list[str]) -> np.ndarray:
    """Median optical-flow magnitude between each consecutive frame pair."""
    size = (config.FLOW_W, config.FLOW_H)

    def load(name):
        img = cv2.imread(str(scene.image_dir / name), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    mags = np.zeros(len(frames) - 1, dtype=np.float64)
    prev = load(frames[0])
    for i in range(1, len(frames)):
        cur = load(frames[i])
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mags[i - 1] = float(np.median(np.linalg.norm(flow, axis=2)))
        prev = cur
    return mags


def _keep_count(mags: np.ndarray, threshold: float) -> int:
    """Simulate keep-when-accumulated-motion-exceeds-threshold selection."""
    kept, acc = 1, 0.0
    for m in mags:
        acc += m
        if acc >= threshold:
            kept += 1
            acc = 0.0
    return kept


def _motion_adaptive(scene: Scene, frames: list[str], budget: int) -> tuple[list[int], float]:
    mags = _flow_magnitudes(scene, frames)
    total = float(mags.sum())
    lo, hi = 1e-6, max(total, 1e-3)
    # binary search threshold so kept count lands near budget
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        kept = _keep_count(mags, mid)
        if kept > budget:
            lo = mid
        else:
            hi = mid
    threshold = hi
    idx, acc = [0], 0.0
    for i, m in enumerate(mags):
        acc += m
        if acc >= threshold:
            idx.append(i + 1)
            acc = 0.0
    if len(idx) > budget:  # thin uniformly over the kept set
        sub = np.linspace(0, len(idx) - 1, budget).astype(int)
        idx = [idx[j] for j in sub]
    return idx, threshold


def select_frames(scene: Scene, force: bool = False) -> tuple[list[str], dict]:
    out_path = scene.workspace / "selected_frames.txt"
    meta_defaults = {"stride": 1, "threshold": None, "budget": None}
    frames = list_frames(scene.image_dir)
    n = len(frames)

    if out_path.exists() and not force:
        names = out_path.read_text().split()
        meta = {"method": "cached", "n_selected": len(names), **meta_defaults}
        return names, meta

    if n <= config.SELECT_ALL_MAX:
        names = frames
        meta = {"method": "all", **meta_defaults}
    elif n <= config.STRIDE_MAX:
        stride = math.ceil(n / config.SELECT_ALL_MAX)
        names = frames[::stride]  # spans full trajectory, never truncated
        meta = {"method": "stride", **meta_defaults, "stride": stride}
    else:
        idx, thr = _motion_adaptive(scene, frames, config.MOTION_BUDGET)
        names = [frames[i] for i in idx]
        # effective stride for overlap computation downstream
        stride = max(1, round(n / max(len(names), 1)))
        meta = {"method": "motion", "stride": stride,
                "threshold": thr, "budget": config.MOTION_BUDGET}

    meta["n_selected"] = len(names)
    scene.workspace.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(names) + "\n")
    return names, meta


def planned_selection(n: int) -> dict:
    """Cheap planning info for the dry-run table (no flow computation)."""
    if n <= config.SELECT_ALL_MAX:
        return {"method": "all", "stride": 1, "n_planned": n}
    if n <= config.STRIDE_MAX:
        stride = math.ceil(n / config.SELECT_ALL_MAX)
        return {"method": "stride", "stride": stride,
                "n_planned": len(range(0, n, stride))}
    stride = max(1, round(n / config.MOTION_BUDGET))
    return {"method": "motion", "stride": stride, "n_planned": config.MOTION_BUDGET}
