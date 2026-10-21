#!/usr/bin/env python
"""Orchestrator for segmentation + depth fusion.

Runs on every scene whose MVS workspace keeps both .geometric.bin and
.photometric.bin depth maps; other scenes are reported as skipped.

  python run_fusion.py                              # all eligible scenes
  python run_fusion.py --scenes a,b
  python run_fusion.py --skip_existing
  python run_fusion.py --array_task_id 0 --n_tasks 4

Needs a GPU (SegFormer).
"""
import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pycolmap
import torch
from PIL import Image

import fusion_config as cfg
import fusion_scenes
import segmentation
import depth_fusion
import depth_io
import mvs_config as mvs_cfg

CSV_FIELDS = ["scene", "mvs_status", "model_used", "n_frames", "n_fused",
              "export_fraction", "mean_valid_fraction_non_sky",
              "median_ground_cutoff", "weak_fraction",
              "fusion_status", "wall_clock_s"]


def load_mvs_info() -> dict:
    """scene -> {model_used, mvs_status} from the MVS stage's own results."""
    out = {}
    if mvs_cfg.MVS_MASTER_CSV.exists():
        with open(mvs_cfg.MVS_MASTER_CSV) as f:
            for r in csv.DictReader(f):
                out[r["scene"]] = {"model_used": r.get("model_used", "full"),
                                   "mvs_status": r.get("mvs_status", "")}
    return out


def load_log(scene) -> dict:
    p = scene.workspace / "fusion_log.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_log(scene, log: dict) -> None:
    with open(scene.workspace / "fusion_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)


def update_summary(row: dict) -> None:
    rows = {}
    if cfg.FUSION_RESULTS_CSV.exists():
        with open(cfg.FUSION_RESULTS_CSV) as f:
            for r in csv.DictReader(f):
                rows[r["scene"]] = r
    rows[row["scene"]] = {k: row.get(k, "") for k in CSV_FIELDS}
    cfg.FUSION_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.FUSION_RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for name in sorted(rows):
            w.writerow(rows[name])


def frame_names_for_scene(scene) -> list[str]:
    d = scene.workspace / "dense" / "stereo" / "depth_maps"
    names = sorted(p.name[:-len(".geometric.bin")]
                   for p in d.glob("*.geometric.bin"))
    return names


def process_scene(scene, model_used: str, mvs_status: str, device: str,
                  batch_size: int, skip_existing: bool, force: bool) -> None:
    log = load_log(scene)
    if skip_existing and not force and log.get("fusion_status"):
        print(f"[skip] {scene.name}: existing {log['fusion_status']}", flush=True)
        return

    t0 = time.time()
    names = frame_names_for_scene(scene)
    log = {"scene": scene.name, "mvs_status": mvs_status,
          "model_used": model_used, "n_frames": len(names)}

    try:
        # Segmentation. ALWAYS cache-reuse existing seg PNGs regardless
        # of --force: forcing a fusion re-run should not also re-run
        # SegFormer inference.
        seg_stats = segmentation.run_segmentation(
            scene, scene.image_dir, names, device, batch_size=batch_size,
            skip_existing=True)
        log["segmentation"] = seg_stats

        # per-scene pose model for the ground-range cutoff
        model_dir_name = "best" if model_used == "best_fallback" else "full"
        rec = pycolmap.Reconstruction(scene.sfm_dir / "sparse" / model_dir_name)
        cam = next(iter(rec.cameras.values()))
        focal_px = float(cam.params[0])
        ground_cutoff = depth_fusion.ground_range_cutoff(rec, focal_px)
        log["ground_cutoff"] = ground_cutoff

        depth_maps_dir = scene.workspace / "dense" / "stereo" / "depth_maps"
        seg_dir = scene.workspace / "seg"
        out_dir = scene.workspace / "fused_depth"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_fused, n_errors, weak_count = 0, 0, 0
        valid_fracs = []
        for name in names:
            out_path = out_dir / f"{name}.h5"
            if skip_existing and not force and out_path.exists():
                n_fused += 1
                continue
            try:
                geo = depth_io.read_depth_bin(depth_maps_dir / f"{name}.geometric.bin")
                photo_path = depth_maps_dir / f"{name}.photometric.bin"
                photo = (depth_io.read_depth_bin(photo_path)
                        if photo_path.exists() else np.zeros_like(geo))
                seg_path = seg_dir / f"{name}.png"
                if not seg_path.exists():
                    raise FileNotFoundError(f"missing segmentation {seg_path}")
                seg = np.array(Image.open(seg_path))

                fused, source, stats = depth_fusion.fuse_frame(
                    geo, photo, seg, ground_cutoff, device)

                with h5py.File(out_path, "w") as f:
                    d = f.create_dataset("depth", data=fused, compression="gzip",
                                         compression_opts=4)
                    f.create_dataset("source", data=source, compression="gzip",
                                     compression_opts=4)
                    d.attrs.update({
                        "valid_fraction_non_sky": stats["valid_fraction_non_sky"],
                        "weak": stats["weak"],
                        "n_photo_rescued": stats["n_photo_rescued"],
                        "n_stability_rejected": stats["n_stability_rejected"],
                        "n_ground_range_masked": stats["n_ground_range_masked"],
                        "n_dynamic_components": stats["n_dynamic_components"],
                        "n_dynamic_kept_px": stats["n_dynamic_kept_px"],
                        "n_dynamic_dropped_px": stats["n_dynamic_dropped_px"],
                        "n_final_valid": stats["n_final_valid"],
                        "ground_cutoff": ground_cutoff,
                        "model_used": model_used,
                        "fusion_version": cfg.FUSION_VERSION,
                    })
                n_fused += 1
                weak_count += int(stats["weak"])
                valid_fracs.append(stats["valid_fraction_non_sky"])
            except Exception as e:  # noqa: BLE001 — per-frame errors never fatal
                n_errors += 1
                print(f"[fusion] fusion error on {name}: {e}", flush=True)

        export_fraction = n_fused / max(len(names), 1)
        weak_fraction = weak_count / max(n_fused, 1)
        mean_valid = float(np.mean(valid_fracs)) if valid_fracs else 0.0

        if export_fraction < cfg.FUSION_GATE_FAIL_EXPORT_FRACTION:
            status = "FUSION_FAIL"
        elif (export_fraction >= cfg.FUSION_GATE_MIN_EXPORT_FRACTION
              and weak_fraction < cfg.FUSION_GATE_MAX_WEAK_FRACTION):
            status = "FUSION_OK"
        else:
            status = "FUSION_WEAK"

        log.update(n_fused=n_fused, n_errors=n_errors,
                   export_fraction=round(export_fraction, 4),
                   mean_valid_fraction_non_sky=round(mean_valid, 4),
                   weak_fraction=round(weak_fraction, 4),
                   fusion_status=status)
    except Exception:  # noqa: BLE001 — a scene must never kill the batch
        tb = traceback.format_exc()
        print(f"[EXCEPTION] {scene.name}:\n{tb}", flush=True)
        log.update(fusion_status="FUSION_FAIL", exception=tb,
                  n_fused=0, export_fraction=0.0,
                  mean_valid_fraction_non_sky=0.0, weak_fraction=1.0)

    log["wall_clock_s"] = round(time.time() - t0, 1)
    save_log(scene, log)
    update_summary({
        "scene": scene.name, "mvs_status": log.get("mvs_status", ""),
        "model_used": model_used, "n_frames": log["n_frames"],
        "n_fused": log.get("n_fused", 0),
        "export_fraction": log.get("export_fraction", ""),
        "mean_valid_fraction_non_sky": log.get("mean_valid_fraction_non_sky", ""),
        "median_ground_cutoff": log.get("ground_cutoff", ""),
        "weak_fraction": log.get("weak_fraction", ""),
        "fusion_status": log["fusion_status"],
        "wall_clock_s": log["wall_clock_s"],
    })
    print(f"[{log['fusion_status']}] {scene.name}: {log.get('n_fused', 0)}/"
          f"{log['n_frames']} fused, mean_valid={log.get('mean_valid_fraction_non_sky')} "
          f"({log['wall_clock_s']}s)",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--array_task_id", type=int, default=None)
    ap.add_argument("--n_tasks", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scenes, skipped = fusion_scenes.load_scenes(
        scenes_filter=args.scenes.split(",") if args.scenes else None)
    print(f"{len(scenes)} eligible scenes (both bin kinds); "
          f"{len(skipped)} skipped (geometric-only, re-run later): {skipped}")

    if args.array_task_id is not None and args.n_tasks:
        scenes = fusion_scenes.array_split(scenes, args.array_task_id, args.n_tasks)
        print(f"Array task {args.array_task_id}/{args.n_tasks}: {len(scenes)} scenes")

    mvs_info = load_mvs_info()
    for s in scenes:
        info = mvs_info.get(s.name, {})
        process_scene(s, info.get("model_used", "full"), info.get("mvs_status", ""),
                     device, args.batch_size, args.skip_existing, args.force)

    print("\nDone. Fusion results:", cfg.FUSION_RESULTS_CSV)


if __name__ == "__main__":
    main()
