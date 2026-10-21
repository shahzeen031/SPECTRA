"""Stage 6 — PnP localization of frames skipped by stage 1.

Every original frame not in selected_frames.txt is:
  1. feature-extracted into the SAME database (same fixed camera, id 1),
  2. matched only against its +-PNP_NEIGHBORS registered temporal neighbors
     (targeted pair list; falls back to a small sequential pass if the
     pair-list binding is unavailable),
  3. registered against the frozen sparse/best model with
     fix_existing_frames=True and intrinsics locked — existing poses/points
     are not re-optimized.

Combined model is written to {workspace}/sparse/full/.
"""
import shutil
from pathlib import Path

import pycolmap

import config
import stage2_features
import stage3_matching
from scene_index import Scene, list_frames


def _build_pairs(all_frames: list[str], selected: set[str],
                 registered: set[str], k: int) -> list[tuple[str, str]]:
    """For each skipped frame, pair with nearest registered selected frames."""
    reg_ordered = [f for f in all_frames if f in registered]
    if not reg_ordered:
        return []
    reg_pos = {f: i for i, f in enumerate(all_frames)}
    reg_indices = [reg_pos[f] for f in reg_ordered]

    pairs = []
    import bisect
    for i, f in enumerate(all_frames):
        if f in selected:
            continue
        j = bisect.bisect_left(reg_indices, i)
        lo, hi = max(0, j - k), min(len(reg_ordered), j + k)
        for r in reg_ordered[lo:hi]:
            pairs.append((f, r))
    return pairs


def run_pnp(scene: Scene, profile: dict, feature_type: str) -> dict:
    workspace = scene.workspace
    best_dir = workspace / "sparse" / "best"
    if not best_dir.exists():
        return {"status": "SKIPPED", "reason": "no sparse/best model"}

    all_frames = list_frames(scene.image_dir)
    selected = set((workspace / "selected_frames.txt").read_text().split())
    skipped = [f for f in all_frames if f not in selected]
    if not skipped:
        # nothing to localize; full == best
        full_dir = workspace / "sparse" / "full"
        if full_dir.exists():
            shutil.rmtree(full_dir)
        shutil.copytree(best_dir, full_dir)
        return {"status": "NOTHING_TO_DO", "attempted": 0, "localized": 0}

    rec = pycolmap.Reconstruction(best_dir)
    registered = {rec.images[i].name for i in rec.reg_image_ids()}
    database = stage2_features.db_path(workspace, feature_type)

    # 1. extract features for skipped frames into the same database/camera
    stage2_features.extract(
        workspace, scene.image_dir, skipped, profile["camera_params"],
        feature_type=feature_type, existing_camera_id=1)

    # 2. targeted matching against registered temporal neighbors
    pairs = _build_pairs(all_frames, selected, registered, config.PNP_NEIGHBORS)
    pairs_file = workspace / "pnp_pairs.txt"
    pairs_file.write_text("\n".join(f"{a} {b}" for a, b in pairs) + "\n")
    used_pairs = stage3_matching.match_pairs_from_file(
        database, pairs_file, feature_type)
    if not used_pairs:  # fallback: cheap sequential pass over full frame set
        stage3_matching.match_sequential(
            database, stride=1, n_frames=0, env="none",
            feature_type=feature_type)

    # 3. register against frozen model
    full_dir = workspace / "sparse" / "full"
    if full_dir.exists():
        shutil.rmtree(full_dir)
    full_dir.mkdir(parents=True)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.fix_existing_frames = True          # freeze existing poses/points
    opts.num_threads = config.NUM_THREADS
    opts.max_runtime_seconds = config.MAPPER_MAX_RUNTIME_S

    recs = pycolmap.incremental_mapping(
        database_path=database, image_path=scene.image_dir,
        output_path=full_dir, options=opts, input_path=best_dir)

    if recs:
        best_idx = max(recs, key=lambda k: recs[k].num_reg_images())
        full_rec = recs[best_idx]
        # rewrite full_dir with only the chosen model
        shutil.rmtree(full_dir)
        full_dir.mkdir(parents=True)
        full_rec.write(full_dir)
        n_full = full_rec.num_reg_images()
    else:
        shutil.rmtree(full_dir)
        shutil.copytree(best_dir, full_dir)
        n_full = len(registered)

    localized = max(0, n_full - len(registered))
    return {
        "status": "OK",
        "attempted": len(skipped),
        "localized": localized,
        "localization_rate": round(localized / max(len(skipped), 1), 4),
        "n_full_registered": n_full,
        "pair_matching": "pairs_file" if used_pairs else "sequential_fallback",
    }
