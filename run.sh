#!/bin/bash
# Runs the whole pipeline: RGB images -> poses (SfM) -> depth maps (MVS) -> RGB-X pair dataset.
#
#   ./run.sh            # every step, in order (re-running resumes, finished scenes are skipped)
#   ./run.sh mvs        # only one step: sfm | mvs | fusion | refine | scene_npz | overlap | pairs | scene_info
#
# Every path below can be overridden from the shell, e.g.  DATA_ROOT=/my/data ./run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

export PYTHON="${PYTHON:-python}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"                    # input: {scene}/rgb, {scene}/x, intrinsics.txt
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output}"              # everything the pipeline writes
export SFM_OUTPUT_ROOT="$OUTPUT_ROOT/sfm"
export MVS_OUTPUT_ROOT="$OUTPUT_ROOT/mvs"
export PAIRS_OUTPUT_ROOT="$OUTPUT_ROOT/image_pairs"
export MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
export COLMAP_CLI="${COLMAP_CLI:-$(command -v colmap || echo colmap)}"   # optional (ALIKED fallback only)

mkdir -p "$SFM_OUTPUT_ROOT" "$MVS_OUTPUT_ROOT" "$PAIRS_OUTPUT_ROOT"

step() {
  echo "================ $1 ================"
  case "$1" in
    sfm)        "$PYTHON" "$ROOT/sfm/run_sfm.py" --stage all --skip_existing ;;
    mvs)        "$PYTHON" "$ROOT/mvs/run_mvs.py" --skip_existing ;;
    fusion)     "$PYTHON" "$ROOT/mvs/run_fusion.py" --skip_existing ;;
    refine)     "$PYTHON" "$ROOT/mvs/export_refined_depth.py" ;;
    scene_npz)  "$PYTHON" "$ROOT/image_pairs/run_scene_npz.py" --skip_existing ;;
    overlap)    "$PYTHON" "$ROOT/image_pairs/compute_depth_overlap.py" ;;
    pairs)      "$PYTHON" "$ROOT/image_pairs/select_pairs.py" ;;
    scene_info) "$PYTHON" "$ROOT/image_pairs/export_scene_info.py" ;;
    *) echo "unknown step: $1 (sfm|mvs|fusion|refine|scene_npz|overlap|pairs|scene_info)"; exit 2 ;;
  esac
}

if [ $# -gt 0 ]; then
  for s in "$@"; do step "$s"; done
else
  for s in sfm mvs fusion refine scene_npz overlap pairs scene_info; do step "$s"; done
fi
echo "Done. Final dataset: $PAIRS_OUTPUT_ROOT/scene_info"
