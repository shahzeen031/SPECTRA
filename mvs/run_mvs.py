#!/usr/bin/env python
"""Orchestrator for the MVS dense-depth stage.

  python run_mvs.py                                  # all eligible scenes
  python run_mvs.py --scenes 1_sunny_data_1_urban
  python run_mvs.py --status WARN
  python run_mvs.py --skip_existing
  python run_mvs.py --array_task_id 0 --n_tasks 4    # split scenes across parallel runs

"""
import argparse
import json
import time
import traceback

import undistort
import patch_match
import depth_io
import mvs_gate
import mvs_config as cfg
import mvs_scenes


def load_log(scene) -> dict:
    p = scene.workspace / "mvs_log.json"
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_log(scene, log: dict) -> None:
    scene.workspace.mkdir(parents=True, exist_ok=True)
    with open(scene.workspace / "mvs_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)


def process_scene(scene, skip_existing: bool, force: bool) -> None:
    log = load_log(scene)
    if skip_existing and not force and log.get("mvs_status"):
        print(f"[skip] {scene.name}: existing {log['mvs_status']}", flush=True)
        return

    t0 = time.time()
    log = {"scene": scene.name, "sfm_status": scene.sfm_status,
           "n_frames": scene.n_frames}
    try:
        # verify + undistort
        und = undistort.run_undistort(scene)
        log["undistort"] = und
        log["mvs_model"] = und["model_used"]
        dense_dir = scene.workspace / "dense"

        # patch-match stereo
        log["stereo"] = patch_match.run_stereo(dense_dir, und["n_images"])

        # export to h5
        export = depth_io.run_export(dense_dir, scene.workspace / "depth_h5",
                                     source_model=und["model_used"])
        log["export"] = {k: export[k] for k in ("n_bins", "n_exported",
                                                "n_empty", "n_errors")}

        # gate
        gate = mvs_gate.evaluate(export, und["n_images"])
        log.update(gate)
        log["mvs_status"] = gate["mvs_status"]

        # Cleanup — the h5 exports are the product, but BOTH raw depth-map
        # kinds are kept for the geo-photo "closer pixel" fusion stage
        # (MegaDepth recipe). Undistorted image copies + normal maps +
        # consistency graphs are pruned (~15 GB/scene) to bound disk use.
        if cfg.CLEANUP_AFTER_EXPORT and gate["mvs_status"] != "MVS_FAIL":
            import shutil
            for sub in ("images", "stereo/normal_maps",
                        "stereo/consistency_graphs"):
                p = dense_dir / sub
                if p.exists():
                    shutil.rmtree(p)
            log["cleanup"] = ("images/normals/consistency pruned; "
                              "depth bins (geo+photo) kept for fusion")
    except Exception:  # noqa: BLE001 — a scene must never kill the batch
        tb = traceback.format_exc()
        print(f"[EXCEPTION] {scene.name}:\n{tb}", flush=True)
        log.update(mvs_status="MVS_FAIL", reason="exception", exception=tb,
                   n_exported=0, export_fraction=0.0,
                   mean_valid_pct=0.0, median_range_ratio=0.0,
                   weak_fraction=1.0)

    log["wall_clock_s"] = round(time.time() - t0, 1)
    save_log(scene, log)
    mvs_gate.update_master_csv({
        "scene": scene.name, "sfm_status": scene.sfm_status,
        "mvs_status": log["mvs_status"],
        "model_used": log.get("mvs_model", ""),
        "n_model_images": (log.get("undistort") or {}).get("n_images", ""),
        "n_depth_maps": log.get("n_exported", 0),
        "export_fraction": log.get("export_fraction", ""),
        "mean_valid_pct": log.get("mean_valid_pct", ""),
        "median_range_ratio": log.get("median_range_ratio", ""),
        "weak_fraction": log.get("weak_fraction", ""),
        "wall_clock_s": log["wall_clock_s"],
    })
    print(f"[{log['mvs_status']}] {scene.name}: "
          f"{log.get('n_exported', 0)} depth maps, "
          f"valid {log.get('mean_valid_pct')}, "
          f"ratio {log.get('median_range_ratio')} "
          f"({log['wall_clock_s']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--status", default=None,
                    help="comma-separated subset of "
                         "EXCELLENT,GOOD,WARN")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--array_task_id", type=int, default=None)
    ap.add_argument("--n_tasks", type=int, default=None)
    args = ap.parse_args()

    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    scenes = mvs_scenes.load_scenes(
        scenes_filter=args.scenes.split(",") if args.scenes else None,
        status_filter=args.status.split(",") if args.status else None)
    if args.array_task_id is not None and args.n_tasks:
        scenes = mvs_scenes.array_split(scenes, args.array_task_id, args.n_tasks)
        print(f"Array task {args.array_task_id}/{args.n_tasks}: "
              f"{len(scenes)} scenes")
    print(f"{len(scenes)} scenes to process "
          f"(sorted by size desc: {[s.name for s in scenes[:3]]}...)")

    for s in scenes:
        process_scene(s, args.skip_existing, args.force)

    print("\nDone. MVS results:", cfg.MVS_MASTER_CSV)


if __name__ == "__main__":
    main()
