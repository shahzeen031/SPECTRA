"""Central configuration for the per-scene COLMAP SfM pipeline."""
import os
from pathlib import Path

# ------------------------------------------------------------------ paths
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parent.parent / "data")))
OUTPUT_ROOT = Path(os.environ.get("SFM_OUTPUT_ROOT", str(Path(__file__).resolve().parent / "output")))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
# pycolmap 4.x needs faiss-format vocab trees (legacy flann .bin hard-aborts).
VOCAB_TREES = {
    "sift": MODELS_DIR / "vocab_tree_faiss_sift.bin",
    "aggressive": MODELS_DIR / "vocab_tree_faiss_sift.bin",
    "aliked": MODELS_DIR / "vocab_tree_faiss_aliked_n16rot.bin",
}

# ALIKED runs through the conda-forge COLMAP CLI (built WITH onnxruntime —
# no pycolmap wheel ships ONNX support). ONNX models are pre-downloaded so
# compute nodes never fetch anything.
COLMAP_CLI = Path(os.environ.get("COLMAP_CLI", "colmap"))
ALIKED_EXTRACT_ONNX = MODELS_DIR / "aliked-n16rot.onnx"
ALIKED_LIGHTGLUE_ONNX = MODELS_DIR / "aliked-lightglue.onnx"

def aliked_available() -> bool:
    return (COLMAP_CLI.exists() and ALIKED_EXTRACT_ONNX.exists()
            and ALIKED_LIGHTGLUE_ONNX.exists()
            and VOCAB_TREES["aliked"].exists())

INVENTORY_CSV = OUTPUT_ROOT / "scene_inventory.csv"
MASTER_CSV = OUTPUT_ROOT / "master_results.csv"
TEXTURE_STATS_JSON = OUTPUT_ROOT / "texture_quartiles.json"

# ------------------------------------------------------------------ camera
# Fixed PINHOLE intrinsics, never optimized. Read from a text file holding one
# line "fx fy cx cy" (pixels, at the resolution of the images in rgb/):
#   {DATA_ROOT}/{scene}/intrinsics.txt   per-scene override, if present
#   {DATA_ROOT}/intrinsics.txt           otherwise, shared by all scenes
INTRINSICS_FILENAME = "intrinsics.txt"

def read_intrinsics(scene_dir: Path) -> list[float]:
    for p in (scene_dir / INTRINSICS_FILENAME, DATA_ROOT / INTRINSICS_FILENAME):
        if p.exists():
            vals = [float(v) for line in p.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                    for v in line.replace(",", " ").split()]
            if len(vals) != 4:
                raise RuntimeError(f"{p}: expected 4 numbers 'fx fy cx cy', got {vals}")
            return vals
    raise RuntimeError(
        f"no {INTRINSICS_FILENAME} in {scene_dir} or {DATA_ROOT} — "
        "write one line 'fx fy cx cy' for your RGB camera")

# ------------------------------------------------------------------ stage 0
MIN_FRAMES = 50                # below this -> TOO_SMALL
TEXTURE_SAMPLE_FRAMES = 20     # frames sampled for Laplacian variance
TEXTURE_DOWNSCALE_W = 480

# ------------------------------------------------------------------ stage 1
SELECT_ALL_MAX = 400           # N <= 400 -> use all frames
STRIDE_MAX = 1200              # 400 < N <= 1200 -> strided
MOTION_BUDGET = 400            # N > 1200 -> motion-adaptive to this budget
FLOW_W, FLOW_H = 405, 231      # optical-flow working resolution

# ------------------------------------------------------------------ stage 2/3
SIFT_MAX_FEATURES = 8192
AGGRESSIVE_PEAK_THRESHOLD = 0.02 / 6.0      # half the SIFT default (more kps)
AUTO_AGGRESSIVE_WEATHERS = {"rainy", "snowy"}   # + bottom texture quartile
BASE_OVERLAP_NUM = 200         # overlap = max(10, 200 // stride)
MIN_OVERLAP = 10
LOOP_DETECTION_MIN_N = 400     # loop detection ON if N > this or env == urban

# ------------------------------------------------------------------ stage 4
ACCEPT_REG_INCREMENTAL = 0.80  # largest / combined incremental threshold
ACCEPT_REG_GLOBAL = 0.70       # global mapping threshold
MAPPER_MAX_RUNTIME_S = 6 * 3600  # per mapping attempt safety cap

# ------------------------------------------------------------------ stage 5 gate
GATE_MIN_REG_RATE = 0.60
GATE_MIN_PTS_PER_IMG = 40
GATE_PASS_REPROJ = 0.75
GATE_WARN_REPROJ = 1.10
EXCELLENT_REG_RATE = 0.95
FOCAL_DRIFT_TOL = 1e-6

# ------------------------------------------------------------------ stage 6
PNP_NEIGHBORS = 5              # +-k registered temporal neighbors to match

# ------------------------------------------------------------------ misc
NUM_THREADS = 8
