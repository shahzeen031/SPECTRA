#!/usr/bin/env python
"""Orchestrator for the per-scene COLMAP SfM pipeline.

Usage examples:
  python run_sfm.py --stage profile                 # stage 0 + plan table
  python run_sfm.py --stage sfm --skip_existing     # stages 1-5
  python run_sfm.py --stage pnp                     # stage 6
  python run_sfm.py --stage all --skip_existing
  python run_sfm.py --stage sfm --scenes 1_sunny_data_1_urban
  python run_sfm.py --stage sfm --weather rainy,snowy
  python run_sfm.py --stage all --array_index 0 --array_total 4

"""
import argparse
import csv
import json
import time
import traceback

import config
import scene_index
import stage0_profile
import stage1_select
import stage2_features
import stage3_matching
import stage4_mapping
import stage5_gate
import stage6_pnp

CSV_FIELDS = [
    "scene", "weather", "env", "n_frames", "n_selected", "select_method",
    "stride", "feature_type", "overlap", "loop_detection", "mapper_used",
    "gate_tier", "reg_rate", "pts_per_img", "reproj_error",
    "pnp_attempted", "pnp_localized", "final_status", "wallclock_s",
]


# --------------------------------------------------------------- logging
def load_log(scene: scene_index.Scene) -> dict:
    p = scene.workspace / "pipeline_log.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_log(scene: scene_index.Scene, log: dict) -> None:
    scene.workspace.mkdir(parents=True, exist_ok=True)
    with open(scene.workspace / "pipeline_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)


def update_master_csv(log: dict) -> None:
    """Incremental per-scene update so partial progress is visible mid-run."""
    rows = {}
    if config.MASTER_CSV.exists():
        with open(config.MASTER_CSV) as f:
            for row in csv.DictReader(f):
                rows[row["scene"]] = row

    gate = log.get("gate", {}) or {}
    sel = log.get("selection", {}) or {}
    match = log.get("matching", {}) or {}
    pnp = log.get("pnp", {}) or {}
    rows[log["scene"]] = {
        "scene": log["scene"],
        "weather": log.get("weather", ""),
        "env": log.get("env", ""),
        "n_frames": log.get("n_frames", ""),
        "n_selected": sel.get("n_selected", ""),
        "select_method": sel.get("method", ""),
        "stride": sel.get("stride", ""),
        "feature_type": log.get("feature_type", ""),
        "overlap": match.get("overlap", ""),
        "loop_detection": match.get("loop_detection", ""),
        "mapper_used": log.get("mapper_used", ""),
        "gate_tier": gate.get("tier", ""),
        "reg_rate": gate.get("reg_rate", ""),
        "pts_per_img": gate.get("pts_per_img", ""),
        "reproj_error": gate.get("reproj_error", ""),
        "pnp_attempted": pnp.get("attempted", ""),
        "pnp_localized": pnp.get("localized", ""),
        "final_status": log.get("final_status", ""),
        "wallclock_s": log.get("wallclock_s", ""),
    }
    with open(config.MASTER_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for name in sorted(rows):
            w.writerow({k: rows[name].get(k, "") for k in CSV_FIELDS})


# --------------------------------------------------------------- planning
def plan_row(profile: dict, texture_q25: float) -> dict:
    n = profile["n_frames"]
    sel = stage1_select.planned_selection(n)
    if profile["too_small"]:
        tier = "TOO_SMALL"
    else:
        tier = sel["method"]
    esc = "ALIKED" if config.aliked_available() else "AGGR-SIFT"
    feature = f"{esc}(auto)" if stage0_profile.wants_auto_aggressive(
        profile, texture_q25) else "SIFT"
    overlap = stage3_matching.overlap_for_stride(sel["stride"])
    loop = (n > config.LOOP_DETECTION_MIN_N) or (profile["env"] == "urban")
    return {
        "scene": profile["scene"], "weather": profile["weather"],
        "env": profile["env"], "N": n, "tier": tier,
        "n_sel": sel["n_planned"] if tier != "TOO_SMALL" else 0,
        "stride": sel["stride"], "texture": round(profile["texture"], 1),
        "feature": feature if tier != "TOO_SMALL" else "-",
        "overlap": overlap if tier != "TOO_SMALL" else "-",
        "loop": ("ON" if loop else "off") if tier != "TOO_SMALL" else "-",
    }


def print_plan(rows: list[dict]) -> None:
    cols = ["scene", "weather", "env", "N", "tier", "n_sel", "stride",
            "texture", "feature", "overlap", "loop"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    n_small = sum(1 for r in rows if r["tier"] == "TOO_SMALL")
    n_auto = sum(1 for r in rows if r["feature"].endswith("(auto)"))
    print(f"\n{len(rows)} scenes | TOO_SMALL: {n_small} | "
          f"auto-escalated: {n_auto} | "
          f"SIFT-first: {len(rows) - n_small - n_auto}")


# --------------------------------------------------------------- sfm core
def run_sfm_once(scene, profile, names, sel_meta, feature_type) -> dict:
    """features -> matching -> mapping -> gate for one feature type."""
    t = {}
    t0 = time.time()
    database = stage2_features.extract(
        scene.workspace, scene.image_dir, names,
        profile["camera_params"], feature_type=feature_type)
    t["features_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    match_info = stage3_matching.match_sequential(
        database, sel_meta["stride"], profile["n_frames"], profile["env"],
        feature_type=feature_type)
    t["matching_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    map_result = stage4_mapping.run_mapping(
        scene.workspace, database, scene.image_dir, len(names),
        tag=feature_type)
    t["mapping_s"] = round(time.time() - t0, 1)

    gate = None
    if map_result.get("best_dir"):
        t0 = time.time()
        gate = stage5_gate.evaluate(
            map_result["best_dir"], len(names), profile["camera_params"])
        t["gate_s"] = round(time.time() - t0, 1)

    return {"feature_type": feature_type, "matching": match_info,
            "mapping": map_result, "gate": gate, "timing": t}


def gate_score(run: dict) -> tuple:
    """Orderable quality score for choosing between feature-tier runs."""
    tier_rank = {"PASS": 2, "WARN": 1, "FAIL": 0, None: -1}
    gate = run.get("gate") or {}
    return (tier_rank.get(gate.get("tier"), -1), gate.get("reg_rate", 0.0))


def process_scene_sfm(scene, texture_q25, skip_existing, force) -> dict:
    log = load_log(scene)
    if skip_existing and not force:
        gate = (log.get("gate") or {})
        if ((scene.workspace / "sparse" / "best").exists()
                and gate.get("tier") in ("PASS", "WARN")):
            print(f"[skip] {scene.name}: existing {gate.get('tier')} model")
            return log

    t_start = time.time()
    profile = stage0_profile.profile_scene(scene)
    log.update(scene=scene.name, weather=profile["weather"],
               env=profile["env"], n_frames=profile["n_frames"])

    if profile["too_small"]:
        log.update(final_status="TOO_SMALL", wallclock_s=0)
        save_log(scene, log)
        update_master_csv(log)
        print(f"[TOO_SMALL] {scene.name}: {profile['n_frames']} frames")
        return log

    names, sel_meta = stage1_select.select_frames(scene, force=force)
    log["selection"] = sel_meta

    # escalation tier: ALIKED via the ONNX-enabled CLI env when present,
    # aggressive SIFT (DSP+affine+guided) otherwise
    esc_ft = "aliked" if config.aliked_available() else "aggressive"

    # feature plan: low-texture rainy/snowy scenes go straight to the
    # escalation tier, everything else SIFT first
    auto_escalate = stage0_profile.wants_auto_aggressive(profile, texture_q25)
    first_ft = esc_ft if auto_escalate else "sift"
    log["auto_escalated"] = auto_escalate
    log["escalation_tier"] = esc_ft

    runs = [run_sfm_once(scene, profile, names, sel_meta, first_ft)]
    best = runs[0]

    # escalate if the plain SIFT run wasn't accepted or failed the gate
    gate_tier = (best.get("gate") or {}).get("tier")
    if first_ft == "sift" and (
            not best["mapping"]["accepted"] or gate_tier == "FAIL"):
        print(f"[escalate] {scene.name}: SIFT run "
              f"(accepted={best['mapping']['accepted']}, gate={gate_tier}) "
              f"-> retrying with {esc_ft}")
        runs.append(run_sfm_once(scene, profile, names, sel_meta, esc_ft))
        if gate_score(runs[1]) > gate_score(runs[0]):
            best = runs[1]

    # promote the winning per-feature model to canonical sparse/best
    if best["mapping"].get("best_dir"):
        stage4_mapping.promote_best(scene.workspace, best["feature_type"])

    log.update(
        feature_type=best["feature_type"],
        matching=best["matching"],
        mapper_used=best["mapping"].get("mapper_used"),
        mapping=best["mapping"],
        gate=best["gate"],
        escalation_runs=[{"feature_type": r["feature_type"],
                          "accepted": r["mapping"]["accepted"],
                          "gate_tier": (r.get("gate") or {}).get("tier"),
                          "timing": r["timing"]} for r in runs],
    )

    gate = best.get("gate") or {}
    if gate.get("tier") == "PASS":
        log["final_status"] = ("EXCELLENT" if gate.get("reg_rate", 0)
                               >= config.EXCELLENT_REG_RATE else "GOOD")
    elif gate.get("tier") == "WARN":
        log["final_status"] = "WARN"
    else:
        log["final_status"] = "SFM_FAIL"

    log["wallclock_s"] = round(time.time() - t_start, 1)
    save_log(scene, log)
    update_master_csv(log)
    print(f"[{log['final_status']}] {scene.name}: "
          f"feature={best['feature_type']} mapper={log['mapper_used']} "
          f"reg_rate={gate.get('reg_rate')} reproj={gate.get('reproj_error')} "
          f"({log['wallclock_s']}s)")
    return log


def process_scene_pnp(scene, force) -> None:
    log = load_log(scene)
    if log.get("final_status") in ("TOO_SMALL", "SFM_FAIL", None):
        print(f"[pnp-skip] {scene.name}: status={log.get('final_status')}")
        return
    if log.get("pnp", {}).get("status") == "OK" and not force:
        print(f"[pnp-skip] {scene.name}: already localized")
        return
    profile = stage0_profile.profile_scene(scene)
    t0 = time.time()
    pnp = stage6_pnp.run_pnp(scene, profile, log.get("feature_type", "sift"))
    pnp["pnp_s"] = round(time.time() - t0, 1)
    log["pnp"] = pnp
    save_log(scene, log)
    update_master_csv(log)
    print(f"[pnp] {scene.name}: localized {pnp.get('localized')}/"
          f"{pnp.get('attempted')} ({pnp['pnp_s']}s)")


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "profile", "sfm", "pnp"])
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scenes", default=None,
                    help="comma-separated canonical scene names")
    ap.add_argument("--weather", default=None,
                    help="comma-separated: sunny,cloudy,rainy,snowy")
    ap.add_argument("--array_index", type=int, default=None)
    ap.add_argument("--array_total", type=int, default=None)
    args = ap.parse_args()

    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # discovery + inventory (always, before anything else)
    all_scenes = scene_index.discover_scenes()
    scene_index.write_inventory(all_scenes)
    print(f"Discovered {len(all_scenes)} scenes "
          f"-> {config.INVENTORY_CSV}")

    scenes = scene_index.filter_scenes(
        all_scenes,
        names=args.scenes.split(",") if args.scenes else None,
        weathers=args.weather.split(",") if args.weather else None)

    # ---------------- stage 0: profile (always over ALL scenes so the
    # texture quartile is global, regardless of filters/array split)
    if args.stage in ("all", "profile"):
        profiles = []
        for s in all_scenes:
            try:
                profiles.append(stage0_profile.profile_scene(s, force=args.force))
            except Exception as e:  # noqa: BLE001
                print(f"[profile-error] {s.name}: {e}")
        stats = stage0_profile.compute_texture_quartiles(profiles)
        print(f"\nTexture q25 = {stats['q25']:.1f} over {stats['n_scenes']} scenes\n")
        prof_by_name = {p["scene"]: p for p in profiles}
        rows = [plan_row(prof_by_name[s.name], stats["q25"])
                for s in scenes if s.name in prof_by_name]
        print_plan(rows)
        if args.stage == "profile":
            return

    # array split (after filtering, deterministic by sorted name)
    if args.array_index is not None and args.array_total:
        scenes = sorted(scenes, key=lambda s: s.name)
        scenes = [s for i, s in enumerate(scenes)
                  if i % args.array_total == args.array_index]
        print(f"Array task {args.array_index}/{args.array_total}: "
              f"{len(scenes)} scenes")

    texture_q25 = stage0_profile.load_texture_quartiles()["q25"]

    if args.stage in ("all", "sfm"):
        for s in scenes:
            try:
                process_scene_sfm(s, texture_q25, args.skip_existing, args.force)
            except Exception:  # noqa: BLE001 — never abort the batch
                tb = traceback.format_exc()
                print(f"[EXCEPTION] {s.name}:\n{tb}")
                log = load_log(s)
                log.update(scene=s.name, weather=s.weather, env=s.env,
                           n_frames=s.n_frames, final_status="SFM_FAIL",
                           exception=tb)
                save_log(s, log)
                update_master_csv(log)

    if args.stage in ("all", "pnp"):
        for s in scenes:
            try:
                process_scene_pnp(s, args.force)
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[PNP-EXCEPTION] {s.name}:\n{tb}")
                log = load_log(s)
                log["pnp"] = {"status": "EXCEPTION", "traceback": tb}
                save_log(s, log)

    print("\nDone. Master results:", config.MASTER_CSV)


if __name__ == "__main__":
    main()
