"""MVS gate — per-scene depth quality gate + incremental mvs_results.csv.

MVS_OK   : exported/model >= 0.8 AND weak fraction < 0.5
MVS_WEAK : complete but majority-weak — keep, flag. For WARN
           scenes this is the training-set-inclusion arbiter, so sfm_status
           is carried into the CSV next to mvs_status.
MVS_FAIL : exported < 50% of model images, or export errors dominated
"""
import csv

import numpy as np

import mvs_config as cfg

CSV_FIELDS = ["scene", "sfm_status", "mvs_status", "model_used", "n_model_images",
              "n_depth_maps", "export_fraction", "mean_valid_pct",
              "median_range_ratio", "weak_fraction", "wall_clock_s"]


def evaluate(export: dict, n_model_images: int) -> dict:
    frames = export["frames"]
    n_exp = len(frames)
    export_fraction = n_exp / max(n_model_images, 1)
    if n_exp:
        mean_valid = float(np.mean([f["valid_pct"] for f in frames]))
        med_ratio = float(np.median([f["range_ratio"] for f in frames]))
        weak_fraction = float(np.mean([f["weak"] for f in frames]))
    else:
        mean_valid = med_ratio = 0.0
        weak_fraction = 1.0

    if export_fraction < cfg.GATE_FAIL_EXPORT_FRACTION:
        status, reason = "MVS_FAIL", (
            f"exported {n_exp}/{n_model_images} "
            f"({100 * export_fraction:.0f}% < 50%)")
    elif (export_fraction >= cfg.GATE_MIN_EXPORT_FRACTION
          and weak_fraction < cfg.GATE_MAX_WEAK_FRACTION):
        status, reason = "MVS_OK", ""
    else:
        status, reason = "MVS_WEAK", (
            f"export_fraction={export_fraction:.2f}, "
            f"weak_fraction={weak_fraction:.2f}")

    return {"mvs_status": status, "reason": reason,
            "n_exported": n_exp, "export_fraction": round(export_fraction, 4),
            "mean_valid_pct": round(mean_valid, 4),
            "median_range_ratio": round(med_ratio, 2),
            "weak_fraction": round(weak_fraction, 4)}


def update_master_csv(row: dict) -> None:
    """Incremental per-scene upsert, same pattern as the SfM master CSV."""
    rows = {}
    if cfg.MVS_MASTER_CSV.exists():
        with open(cfg.MVS_MASTER_CSV) as f:
            for r in csv.DictReader(f):
                rows[r["scene"]] = r
    rows[row["scene"]] = {k: row.get(k, "") for k in CSV_FIELDS}
    cfg.MVS_MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.MVS_MASTER_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for name in sorted(rows):
            w.writerow(rows[name])
