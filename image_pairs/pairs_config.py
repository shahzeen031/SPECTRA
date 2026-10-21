"""Config for the per-scene RGB->X NPZ stage.

Scope: only scenes with a completed fusion pass (FUSION_OK/WEAK) —
every pair's depth_paths must point at a real fused depth h5, so scenes
without one are excluded here (not deferred-with-empty-path).

Overlap is computed by simple sparse-point reprojection (SfM points3D
projected between camera pairs), NOT the dense/fused depth maps — cheap,
CPU-only, and independent of the MVS stage's own coverage.
"""
import os
import sys
from pathlib import Path

# reuse the MVS stage configs (output roots, fusion results csv)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mvs"))

import mvs_config as mvs_cfg
import fusion_config as fusion_cfg

SFM_OUTPUT_ROOT = mvs_cfg.SFM_OUTPUT_ROOT
MVS_OUTPUT_ROOT = mvs_cfg.OUTPUT_ROOT
FUSION_RESULTS_CSV = fusion_cfg.FUSION_RESULTS_CSV

PAIRS_OUTPUT_ROOT = Path(os.environ.get("PAIRS_OUTPUT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "image_pairs")))
OUTPUT_DIR = PAIRS_OUTPUT_ROOT / "scene_npz"
SUMMARY_JSON = OUTPUT_DIR / "conversion_summary.json"

MIN_OVERLAP = 0.1
NUM_THREADS = 8
