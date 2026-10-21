"""Configuration for the segmentation + depth-fusion stages.

Applies to scenes whose MVS workspace keeps BOTH .geometric.bin and
.photometric.bin depth maps.
"""
import os
from pathlib import Path

import mvs_config as mvs_cfg

OUTPUT_ROOT = mvs_cfg.OUTPUT_ROOT          # reuse mvs/output/{scene}/
FUSION_RESULTS_CSV = mvs_cfg.OUTPUT_ROOT / "fusion_results.csv"

# ------------------------------------------------------------------ models
SEGFORMER_MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
SEGFORMER_INPUT_SIZE = 1024          # square resize for the segformer forward pass

# ------------------------------------------------------------------ Cityscapes classes (19)
CS_ROAD, CS_SIDEWALK, CS_BUILDING, CS_WALL, CS_FENCE, CS_POLE = 0, 1, 2, 3, 4, 5
CS_TRAFFIC_LIGHT, CS_TRAFFIC_SIGN, CS_VEGETATION, CS_TERRAIN, CS_SKY = 6, 7, 8, 9, 10
CS_PERSON, CS_RIDER, CS_CAR, CS_TRUCK, CS_BUS = 11, 12, 13, 14, 15
CS_TRAIN, CS_MOTORCYCLE, CS_BICYCLE = 16, 17, 18

SKY_CLASSES = {CS_SKY}
# ground classes: kept where MVS-valid & within reliable range, masked otherwise
GROUND_CLASSES = {CS_ROAD, CS_SIDEWALK, CS_TERRAIN}
# dynamic-candidate classes: per-connected-component evidence test decides
DYNAMIC_CLASSES = {CS_PERSON, CS_RIDER, CS_CAR, CS_TRUCK, CS_BUS, CS_TRAIN,
                   CS_MOTORCYCLE, CS_BICYCLE}
# always-kept static structure: building, wall, fence, pole, traffic light/sign, vegetation
STATIC_CLASSES = {CS_BUILDING, CS_WALL, CS_FENCE, CS_POLE, CS_TRAFFIC_LIGHT,
                  CS_TRAFFIC_SIGN, CS_VEGETATION}

# ------------------------------------------------------------------ fusion thresholds
GEO_PHOTO_RATIO = 1.1          # geo/photo > this -> replace geo with photo (MegaDepth)
STABILITY_RATIO = 1.1          # median-filtered/original ratio > this -> reject pixel
MEDIAN_FILTER_SIZE = 7
MEDIAN_FILTER_PASSES = 2

COMPONENT_EVIDENCE_MIN_VALID = 0.5   # dynamic-class component kept if >= this fraction MVS-valid
COMPONENT_MIN_SIZE = 50              # ignore tiny/noisy segmentation blobs below this pixel count

# ground-class reliable-range cutoff: depth beyond
# k * baseline * focal / MIN_DISPARITY_PX is treated as unreliable triangulation
MIN_DISPARITY_PX = 2.0
GROUND_RANGE_SAFETY = 1.0            # multiplier on the theoretical cutoff

NON_SKY_VALID_FRACTION_WEAK = 0.35   # MegaDepth's own bar; below this -> weak flag

# ------------------------------------------------------------------ scene gate
# mirrors the MVS gate in mvs_config — same philosophy, applied to the fused output
FUSION_GATE_MIN_EXPORT_FRACTION = 0.8
FUSION_GATE_MAX_WEAK_FRACTION = 0.5
FUSION_GATE_FAIL_EXPORT_FRACTION = 0.5

# ------------------------------------------------------------------ misc
NUM_THREADS = 8
FUSION_VERSION = "fusion_v1"
