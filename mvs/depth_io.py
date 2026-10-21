"""Depth I/O — read COLMAP .bin depth maps and export .geometric.bin depth maps to per-frame h5 files.

Reader: pycolmap.DepthMap.read() (version-safe; ported from
dense_pipeline/colmap_dense_pipeline.py), with a manual ampersand-header
parser as fallback (float32, Fortran order, channel 0, transpose).
Per frame: p1/p99 outlier clip -> stats -> weak flag -> gzip-4 h5.
"""
from pathlib import Path

import h5py
import numpy as np
import pycolmap

import mvs_config as cfg


def read_depth_bin(path: Path) -> np.ndarray:
    """float32 (H, W), 0 = invalid."""
    try:
        dm = pycolmap.DepthMap()
        dm.read(str(path))
        return dm.to_array().astype(np.float32).squeeze()
    except Exception:  # noqa: BLE001 — fallback: manual parse
        with open(path, "rb") as f:
            header = b""
            while header.count(b"&") < 3:
                header += f.read(1)
            w, h, c = (int(x) for x in header.decode().split("&")[:3])
            data = np.fromfile(f, dtype=np.float32)
        arr = data.reshape((w, h, c), order="F")
        return np.transpose(arr, (1, 0, 2))[:, :, 0].astype(np.float32)


def export_frame(bin_path: Path, out_dir: Path, source_model: str) -> dict | None:
    """Export one depth map. Returns stats dict, or None if no valid depth."""
    depth = read_depth_bin(bin_path)
    valid = depth > 0
    if not valid.any():
        return None

    v = depth[valid]
    lo, hi = np.percentile(v, [cfg.CLIP_LO_PCT, cfg.CLIP_HI_PCT])
    depth = np.where((depth >= lo) & (depth <= hi), depth, 0.0).astype(np.float32)
    valid = depth > 0
    if not valid.any():
        return None
    v = depth[valid]

    p5, p50, p95, p99 = np.percentile(v, [5, 50, 95, 99])
    valid_pct = float(valid.mean())
    range_ratio = float(p99 / max(p5, 1e-9))
    weak = bool(range_ratio < cfg.WEAK_RANGE_RATIO
                or valid_pct < cfg.WEAK_VALID_PCT)

    image_name = bin_path.name.replace(".geometric.bin", "")
    out_path = out_dir / f"{image_name}.h5"
    with h5py.File(out_path, "w") as f:
        d = f.create_dataset("depth", data=depth, compression="gzip",
                             compression_opts=4)
        d.attrs.update({
            "valid_pct": valid_pct, "p5": float(p5), "p50": float(p50),
            "p95": float(p95), "p99": float(p99), "range_ratio": range_ratio,
            "weak": weak, "source_model": source_model,
            "mvs_version": cfg.MVS_VERSION,
        })
    return {"image": image_name, "valid_pct": valid_pct,
            "range_ratio": range_ratio, "weak": weak}


def run_export(dense_dir: Path, out_dir: Path, source_model: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = sorted((dense_dir / "stereo" / "depth_maps").glob("*.geometric.bin"))
    frames, errors, empty = [], 0, 0
    for b in bins:
        try:
            r = export_frame(b, out_dir, source_model)
            if r is None:
                empty += 1
            else:
                frames.append(r)
        except Exception as e:  # noqa: BLE001 — per-frame errors never fatal
            errors += 1
            print(f"[export] error {b.name}: {e}", flush=True)
    return {"n_bins": len(bins), "n_exported": len(frames),
            "n_empty": empty, "n_errors": errors, "frames": frames}
