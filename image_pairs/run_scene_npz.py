#!/usr/bin/env python
"""Orchestrator: build the per-scene RGB->X NPZ for each fused scene.

  python run_scene_npz.py                    # all fused scenes
  python run_scene_npz.py --scenes a,b
  python run_scene_npz.py --skip_existing

CPU-only (pycolmap + numpy).
"""
import argparse
import csv
import json
import time
import traceback

import build_scene_npz as builder
import pairs_config as cfg
import mvs_scenes


def load_fused_scenes(scenes_filter=None):
    """scene -> (model_used, image_dir, mvs_workspace) for FUSION_OK/WEAK scenes."""
    fusion_info = {}
    with open(cfg.FUSION_RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            if r["fusion_status"] in ("FUSION_OK", "FUSION_WEAK"):
                fusion_info[r["scene"]] = r["model_used"]

    all_scenes = {s.name: s for s in mvs_scenes.load_scenes(
        status_filter=["EXCELLENT", "GOOD"])}

    out = []
    for name, model_used in fusion_info.items():
        if scenes_filter and name not in scenes_filter:
            continue
        s = all_scenes[name]
        out.append((name, model_used, s.image_dir, s.workspace))
    out.sort(key=lambda t: t[0])
    return out


def load_summary() -> dict:
    if cfg.SUMMARY_JSON.exists():
        try:
            return json.load(open(cfg.SUMMARY_JSON))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_summary(summary: dict) -> None:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(cfg.SUMMARY_JSON, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scenes = load_fused_scenes(args.scenes.split(",") if args.scenes else None)
    print(f"{len(scenes)} fused scenes to process")

    summary = load_summary()
    for name, model_used, image_dir, workspace in scenes:
        out_path = cfg.OUTPUT_DIR / f"{name}.npz"
        if args.skip_existing and not args.force and out_path.exists():
            print(f"[skip] {name}")
            continue
        t0 = time.time()
        try:
            stats = builder.build_scene_npz(name, model_used, image_dir, workspace)
            stats["wall_clock_s"] = round(time.time() - t0, 1)
            print(f"[OK] {name}: {stats['n_images_kept']}/{stats['n_images_model']} "
                  f"images, {stats['n_x_available']} with x, "
                  f"{stats['n_pairs']} pairs (mean ov {stats['overlap_mean']:.3f}) "
                  f"({stats['wall_clock_s']}s)", flush=True)
        except Exception:  # noqa: BLE001 — one scene never kills the batch
            tb = traceback.format_exc()
            print(f"[ERROR] {name}:\n{tb}", flush=True)
            stats = {"scene": name, "error": tb,
                    "wall_clock_s": round(time.time() - t0, 1)}
        summary[name] = stats
        save_summary(summary)

    print("\nDone. Summary:", cfg.SUMMARY_JSON)


if __name__ == "__main__":
    main()
