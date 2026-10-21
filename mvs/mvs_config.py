"""Configuration for the MVS dense-depth stage."""
import os
from pathlib import Path

# ------------------------------------------------------------------ inputs
SFM_OUTPUT_ROOT = Path(os.environ.get("SFM_OUTPUT_ROOT", str(Path(__file__).resolve().parent.parent / "sfm" / "output")))
SFM_MASTER_CSV = SFM_OUTPUT_ROOT / "master_results.csv"

# ------------------------------------------------------------------ outputs
OUTPUT_ROOT = Path(os.environ.get("MVS_OUTPUT_ROOT", str(Path(__file__).resolve().parent / "output")))
MVS_MASTER_CSV = OUTPUT_ROOT / "mvs_results.csv"
RATE_JSON = OUTPUT_ROOT / "stereo_rate.json"   # running s/image for ETA

# ------------------------------------------------------------------ binaries
# Patch-match runs via the CLI subprocess so CUDA OOM is a catchable return
# code (never an uncatchable in-process abort). This env has the full CLI.
COLMAP_CLI = Path(os.environ.get("COLMAP_CLI", "colmap"))

# ------------------------------------------------------------------ camera
# Locked PINHOLE intrinsics — the SfM stage reads them from intrinsics.txt and
# records them in sfm_output/{scene}/profile.json; undistort.py checks the model against
# that record (never optimized).
FOCAL_DRIFT_TOL = 1e-6

# ------------------------------------------------------------------ selection
ALLOWED_STATUS = {"EXCELLENT", "GOOD", "WARN"}

# ------------------------------------------------------------------ stereo
GEOM_CONSISTENCY = True
CACHE_SIZE_GB = 16
LOW_VRAM_GB = 16            # below this -> reduce max_image_size
LOW_VRAM_MAX_IMAGE_SIZE = 1600
CHUNK_SIZE = 800            # ref-image chunk size for the OOM retry path
STEREO_TIMEOUT_S = 60 * 3600
DEFAULT_SEC_PER_IMAGE = 4.0  # initial ETA guess, refined by measured rate

# ------------------------------------------------------------------ depth export
CLIP_LO_PCT = 1.0           # zero depths below per-frame p1
CLIP_HI_PCT = 99.0          # and above p99
WEAK_RANGE_RATIO = 3.0      # weak if p99/max(p5,eps) < this ...
WEAK_VALID_PCT = 0.30       # ... OR valid fraction < this (full-image denom)
MVS_VERSION = "1.0"

# ------------------------------------------------------------------ gate
GATE_MIN_EXPORT_FRACTION = 0.8   # exported/model images for MVS_OK
GATE_MAX_WEAK_FRACTION = 0.5     # weak/exported for MVS_OK
GATE_FAIL_EXPORT_FRACTION = 0.5  # below this -> MVS_FAIL

NUM_THREADS = 8

# cleanup
CLEANUP_AFTER_EXPORT = True   # prune dense workspace after successful h5 export
