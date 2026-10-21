"""Stage 4 — mapping with escalation ladder.

Ladder (per feature type):
  1. pycolmap.incremental_mapping, fixed intrinsics.
     Largest model registers >= 80% of selected -> accept.
  2. pycolmap.global_mapping (GLOMAP built into pycolmap >= 3.12) on the same
     database. Accept if >= 70% registered.
     NOTE: pycolmap 4.x exposes no `model_merger` binding; global mapping
     subsumes the merge rung for fragmented incremental models.
  3. If plain SIFT was used and the aggressive tier (DSP-SIFT + affine +
     guided matching) not yet tried: redo features+matching and repeat 1-2
     once (driven by the orchestrator).
  4. Else SFM_FAIL, retain best partial model.

The accepted model is written explicitly to {workspace}/sparse/best/ —
downstream never relies on sparse/0.
"""
import shutil
from pathlib import Path

import pycolmap

import config


def _fix_intrinsics_incremental(opts: pycolmap.IncrementalPipelineOptions) -> None:
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False


def _fix_intrinsics_global(opts: pycolmap.GlobalPipelineOptions) -> None:
    """GLOMAP option names vary across builds — walk candidates defensively."""
    candidates = ("optimize_intrinsics", "refine_focal_length",
                  "refine_principal_point", "refine_extra_params",
                  "ba_refine_focal_length", "ba_refine_principal_point",
                  "ba_refine_extra_params")
    for holder in (opts.mapper,
                   getattr(opts.mapper, "bundle_adjustment", None),
                   getattr(opts.mapper, "global_positioning", None)):
        if holder is None:
            continue
        for attr in candidates:
            if hasattr(holder, attr):
                try:
                    setattr(holder, attr, False)
                except (AttributeError, TypeError):
                    pass


def _reg_stats(recs: dict, n_selected: int) -> dict:
    if not recs:
        return {"largest": 0, "combined": 0, "largest_rate": 0.0,
                "combined_rate": 0.0, "n_models": 0, "best_idx": None}
    counts = {k: r.num_reg_images() for k, r in recs.items()}
    best_idx = max(counts, key=counts.get)
    largest = counts[best_idx]
    combined = sum(counts.values())
    return {"largest": largest, "combined": combined,
            "largest_rate": largest / max(n_selected, 1),
            "combined_rate": combined / max(n_selected, 1),
            "n_models": len(recs), "best_idx": best_idx}


def _write_best(rec: pycolmap.Reconstruction, workspace: Path,
                tag: str) -> Path:
    best_dir = workspace / "sparse" / f"best_{tag}"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True)
    rec.write(best_dir)
    return best_dir


def promote_best(workspace: Path, tag: str) -> Path:
    """Copy the winning per-feature model to the canonical sparse/best."""
    src = workspace / "sparse" / f"best_{tag}"
    dst = workspace / "sparse" / "best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def run_mapping(workspace: Path, database: Path, image_dir: Path,
                n_selected: int, tag: str = "sift") -> dict:
    """Run incremental -> global ladder on one database.

    Returns dict with mapper_used, stats, accepted flag; writes the accepted
    (or best partial) model to sparse/best_{tag}. The orchestrator promotes
    the winning tag to sparse/best.
    """
    result = {"mapper_used": None, "accepted": False, "attempts": []}
    best_rec, best_count, best_mapper = None, -1, None

    # ---- rung 1: incremental ------------------------------------------
    inc_dir = workspace / "sparse" / f"incremental_{tag}"
    inc_dir.mkdir(parents=True, exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    _fix_intrinsics_incremental(opts)
    opts.num_threads = config.NUM_THREADS
    opts.max_runtime_seconds = config.MAPPER_MAX_RUNTIME_S
    recs = pycolmap.incremental_mapping(
        database_path=database, image_path=image_dir,
        output_path=inc_dir, options=opts)
    stats = _reg_stats(recs, n_selected)
    result["attempts"].append({"mapper": "incremental", **{
        k: stats[k] for k in ("largest", "combined", "largest_rate",
                              "combined_rate", "n_models")}})
    if stats["largest"] > best_count and recs:
        best_rec, best_count = recs[stats["best_idx"]], stats["largest"]
        best_mapper = "incremental"
    if stats["largest_rate"] >= config.ACCEPT_REG_INCREMENTAL:
        result.update(mapper_used="incremental", accepted=True)
        result["best_dir"] = str(_write_best(best_rec, workspace, tag))
        result["reg_images"] = best_count
        return result

    # ---- rung 2: global mapping (GLOMAP) ------------------------------
    glob_dir = workspace / "sparse" / f"global_{tag}"
    glob_dir.mkdir(parents=True, exist_ok=True)
    gopts = pycolmap.GlobalPipelineOptions()
    _fix_intrinsics_global(gopts)
    gopts.num_threads = config.NUM_THREADS
    try:
        grecs = pycolmap.global_mapping(
            database_path=database, image_path=image_dir,
            output_path=glob_dir, options=gopts)
    except Exception as e:  # noqa: BLE001 — one scene must never kill the batch
        grecs = {}
        result["global_error"] = repr(e)
    gstats = _reg_stats(grecs, n_selected)
    result["attempts"].append({"mapper": "global", **{
        k: gstats[k] for k in ("largest", "combined", "largest_rate",
                               "combined_rate", "n_models")}})
    if gstats["largest"] > best_count and grecs:
        best_rec, best_count = grecs[gstats["best_idx"]], gstats["largest"]
        best_mapper = "global"
    if gstats["largest_rate"] >= config.ACCEPT_REG_GLOBAL:
        result.update(mapper_used="global", accepted=True)
        result["best_dir"] = str(_write_best(best_rec, workspace, tag))
        result["reg_images"] = best_count
        return result

    # ---- no rung accepted: keep best partial ---------------------------
    result["mapper_used"] = best_mapper
    result["reg_images"] = max(best_count, 0)
    if best_rec is not None:
        result["best_dir"] = str(_write_best(best_rec, workspace, tag))
    return result
