"""Stage 3 — sequential matching with optional vocab-tree loop detection.

overlap = max(10, 200 // stride); quadratic overlap ON.
Loop detection ON for N > 400 or env == urban. Per-tier configuration:

  sift        pycolmap SIFT_BRUTEFORCE, faiss SIFT vocab tree
  aliked      ALIKED_LIGHTGLUE via the ONNX-enabled COLMAP CLI subprocess,
              faiss ALIKED_N16ROT vocab tree
  aggressive  pycolmap SIFT_BRUTEFORCE + guided matching, SIFT vocab tree
"""
import subprocess
from pathlib import Path

import pycolmap

import config


def matcher_options(feature_type: str) -> pycolmap.FeatureMatchingOptions:
    opts = pycolmap.FeatureMatchingOptions()
    opts.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    if feature_type == "aggressive":
        opts.guided_matching = True
    opts.num_threads = config.NUM_THREADS
    return opts


def overlap_for_stride(stride: int) -> int:
    return max(config.MIN_OVERLAP, config.BASE_OVERLAP_NUM // max(stride, 1))


def _match_sequential_aliked_cli(database: Path, overlap: int,
                                 loop_detection: bool) -> None:
    """ALIKED_LIGHTGLUE sequential matching via the ONNX-enabled COLMAP CLI."""
    cmd = [
        str(config.COLMAP_CLI), "sequential_matcher",
        "--database_path", str(database),
        "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
        "--FeatureMatching.num_threads", str(config.NUM_THREADS),
        "--AlikedMatching.lightglue_model_path",
        str(config.ALIKED_LIGHTGLUE_ONNX),
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.quadratic_overlap", "1",
    ]
    if loop_detection:
        cmd += ["--SequentialMatching.loop_detection", "1",
                "--SequentialMatching.vocab_tree_path",
                str(config.VOCAB_TREES["aliked"])]
    for use_gpu in ("1", "0"):
        proc = subprocess.run(
            cmd + ["--FeatureMatching.use_gpu", use_gpu],
            capture_output=True, text=True, errors="replace", timeout=12 * 3600)
        if proc.returncode == 0:
            return
        if use_gpu == "1":
            print(f"[aliked] GPU matching failed (rc={proc.returncode}), "
                  "retrying on CPU")
    raise RuntimeError(
        f"ALIKED CLI matching failed:\n{proc.stderr[-2000:]}")


def match_sequential(database: Path, stride: int, n_frames: int, env: str,
                     feature_type: str = "sift") -> dict:
    overlap = overlap_for_stride(stride)
    loop_detection = (n_frames > config.LOOP_DETECTION_MIN_N) or (env == "urban")
    tree = config.VOCAB_TREES["aliked" if feature_type == "aliked" else feature_type]
    if loop_detection and not tree.exists():
        print(f"[matching] vocab tree {tree} not found — loop detection OFF")
        loop_detection = False

    if feature_type == "aliked":
        _match_sequential_aliked_cli(database, overlap, loop_detection)
        return {"overlap": overlap, "quadratic_overlap": True,
                "loop_detection": loop_detection}

    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pairing.quadratic_overlap = True
    pairing.num_threads = config.NUM_THREADS
    if loop_detection:
        pairing.loop_detection = True
        pairing.vocab_tree_path = str(tree)

    pycolmap.match_sequential(
        database_path=database,
        matching_options=matcher_options(feature_type),
        pairing_options=pairing,
    )
    return {
        "overlap": overlap,
        "quadratic_overlap": True,
        "loop_detection": loop_detection,
    }


def match_pairs_from_file(database: Path, pairs_file: Path,
                          feature_type: str = "sift") -> bool:
    """Targeted pair matching (stage 6). Returns False if the binding lacks
    a match-list option so the caller can fall back to sequential matching."""
    if feature_type == "aliked":
        cmd = [
            str(config.COLMAP_CLI), "matches_importer",
            "--database_path", str(database),
            "--match_list_path", str(pairs_file),
            "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
            "--FeatureMatching.num_threads", str(config.NUM_THREADS),
            "--AlikedMatching.lightglue_model_path",
            str(config.ALIKED_LIGHTGLUE_ONNX),
        ]
        for use_gpu in ("1", "0"):
            proc = subprocess.run(
                cmd + ["--FeatureMatching.use_gpu", use_gpu],
                capture_output=True, text=True, errors="replace", timeout=12 * 3600)
            if proc.returncode == 0:
                return True
        raise RuntimeError(
            f"ALIKED CLI pair matching failed:\n{proc.stderr[-2000:]}")

    pairing = pycolmap.ImportedPairingOptions()
    if not hasattr(pairing, "match_list_path"):
        return False
    pairing.match_list_path = str(pairs_file)
    pycolmap.match_image_pairs(
        database_path=database,
        matching_options=matcher_options(feature_type),
        pairing_options=pairing,
    )
    return True
