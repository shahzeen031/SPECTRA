"""Stage 2 — feature extraction. Three tiers, one database per tier:

  sift        pycolmap SIFT-GPU (default)                 -> database_sift.db
  aliked      ALIKED_N16ROT via the conda-forge COLMAP    -> database_aliked.db
              CLI subprocess (the only ONNX-enabled build;
              pycolmap wheels lack ONNX and hard-abort)
  aggressive  DSP-SIFT + affine + peak/2, CPU (fallback   -> database_aggressive.db
              escalation when the ALIKED CLI env is absent)

Single shared PINHOLE camera per scene with fixed params. Running ALIKED in a
subprocess also isolates any native crash from the batch process.
"""
import subprocess
from pathlib import Path

import pycolmap

import config


def db_path(workspace: Path, feature_type: str) -> Path:
    return workspace / f"database_{feature_type}.db"


def _reader_options(camera_params: list[float]) -> pycolmap.ImageReaderOptions:
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = "PINHOLE"
    opts.camera_params = ",".join(f"{p:.10g}" for p in camera_params)
    return opts


def _extract_aliked_cli(workspace: Path, image_dir: Path,
                        image_names: list[str],
                        camera_params: list[float]) -> Path:
    """ALIKED extraction via the ONNX-enabled COLMAP CLI (GPU with CPU retry)."""
    database = db_path(workspace, "aliked")
    list_file = workspace / "aliked_image_list.txt"
    list_file.write_text("\n".join(image_names) + "\n")
    base = [
        str(config.COLMAP_CLI), "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(image_dir),
        "--image_list_path", str(list_file),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params",
        ",".join(f"{p:.10g}" for p in camera_params),
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.type", "ALIKED_N16ROT",
        "--FeatureExtraction.num_threads", str(config.NUM_THREADS),
        "--AlikedExtraction.max_num_features", str(config.SIFT_MAX_FEATURES),
        "--AlikedExtraction.n16rot_model_path", str(config.ALIKED_EXTRACT_ONNX),
    ]
    for use_gpu in ("1", "0"):  # CPU retry if the CUDA EP fails on this node
        proc = subprocess.run(
            base + ["--FeatureExtraction.use_gpu", use_gpu],
            capture_output=True, text=True, errors="replace", timeout=4 * 3600)
        if proc.returncode == 0:
            return database
        if use_gpu == "1":
            print(f"[aliked] GPU extraction failed (rc={proc.returncode}), "
                  "retrying on CPU")
    raise RuntimeError(
        f"ALIKED CLI extraction failed:\n{proc.stderr[-2000:]}")


def extract(workspace: Path, image_dir: Path, image_names: list[str],
            camera_params: list[float], feature_type: str = "sift",
            existing_camera_id: int = -1) -> Path:
    """Extract features for image_names into the per-type database.

    existing_camera_id: pass the scene camera id (1) when adding more frames
    to an existing database (stage 6) so all frames share one camera.
    """
    if feature_type == "aliked":
        return _extract_aliked_cli(workspace, image_dir, image_names,
                                   camera_params)

    database = db_path(workspace, feature_type)

    opts = pycolmap.FeatureExtractionOptions()
    opts.type = pycolmap.FeatureExtractorType.SIFT
    opts.sift.max_num_features = config.SIFT_MAX_FEATURES
    if feature_type == "aggressive":
        opts.sift.estimate_affine_shape = True      # covariant detector (CPU)
        opts.sift.domain_size_pooling = True        # DSP-SIFT descriptors
        opts.sift.peak_threshold = config.AGGRESSIVE_PEAK_THRESHOLD
        opts.use_gpu = False
    opts.num_threads = config.NUM_THREADS

    reader = _reader_options(camera_params)
    if existing_camera_id > 0:
        reader.existing_camera_id = existing_camera_id

    pycolmap.extract_features(
        database_path=database,
        image_path=image_dir,
        image_names=list(image_names),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=opts,
    )
    return database
