<p align="center">
  <h1 align="center"><ins>SPECTRA</ins>:<br> Cross-Modality Multi-View Image Matching Dataset<br> in Low-Parallax Driving Scenes<br> ⭐WACV 2027⭐</h1>
  <p align="center">
    Shahzeen Ijaz Ahmad
    ·
    Hasan Fehmi Ates
  </p>
  <h2 align="center"><p>
    Paper | Project Page | Dataset | Checkpoints
  </p></h2>
  <p align="center"><em>Links coming soon</em></p>
</p>
<br/>

This repository contains the reconstruction pipeline used to build the SPECTRA
dataset. It turns image sequences from an RGB camera plus a pixel-aligned second
modality (X: thermal, SWIR, NIR, ...) into camera poses, depth maps and verified
RGB↔X training pairs. Poses and depth are computed from the **RGB images only**;
the X images reuse the poses and depth of their matching RGB frames.

## Abstract

In this study we present, to the best of our knowledge, the first real-sensor
RGB-X multi-view, multi-modality image matching dataset for driving scenes.
Matching images across sensor modalities is a prerequisite for perception
systems that pair an RGB camera with a thermal or infrared one, yet it is
evaluated almost entirely on wide-baseline, static, clear-weather benchmarks.
Driving violates all three assumptions, and method rankings obtained under clear
conditions reverse under rain and snow. Existing resources do not close this
gap: pre-aligned collections offer no viewpoint change, synthetic
cross-modality data inherits landmark viewpoints, and driving multi-sensor
datasets release inertial poses rather than matching supervision. Nor can such
supervision simply be reconstructed: Structure-from-Motion followed by
multi-view stereo becomes ill-conditioned under near-parallel forward motion,
dynamic objects and low-texture weather. We therefore build a reconstruction
pipeline for this setting. Intrinsics are fixed from the sensor specification
and never bundle-adjusted, dynamic content is removed by geometric consistency,
and reconstruction escalates adaptively on low-texture scenes under explicit
quality gating, so that unreliable scenes are excluded rather than released as
noisy ground truth. Released poses are verified against two independent
matchers (median rotation error 0.094°). Applied to RGB-SWIR driving imagery,
the pipeline yields SPECTRA: 384,550 verified cross-modality pairs with poses
and depth. Fine-tuning on SPECTRA improves every architecture tested on unseen
RGB-thermal data, across two platforms and two metrics, with margins that grow
as conditions degrade.

## 1. Environment

Needs Linux, an NVIDIA GPU with a CUDA 12 driver, and Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The segmentation model (SegFormer) downloads by itself on the first run, so
that run needs internet access.

Optional: for loop closure on long sequences, put `vocab_tree_faiss_sift.bin`
(from the [COLMAP releases page](https://github.com/colmap/colmap/releases))
in `models/`. Without it, loop detection is turned off.

## 2. Put your data in `data/`

> **Your RGB and X images must be pre-aligned before you run the pipeline.**
> The pipeline does not align them. Each X image must have the same
> resolution as its RGB image, and every pixel must show the same point of the
> scene in both images. For example, warp or rectify your X images into the
> RGB camera's view first. This is required because the X images reuse the
> RGB camera's poses, intrinsics and depth maps.

```
data/
├── intrinsics.txt          <- your RGB camera intrinsics
├── scene_a/
│   ├── rgb/                <- RGB images: 000000.jpg, 000001.jpg, ...
│   └── x/                  <- X images:   000000.jpg, 000001.jpg, ...
└── scene_b/
    ├── rgb/
    └── x/
```

- **One folder per scene.** A scene is one continuous image sequence of at
  least 50 frames, all at the same resolution (`.jpg` or `.png`).
- **`rgb/` and `x/`** hold the same frames, pre-aligned as described above.
  Frames are matched by the number in the file name, so
  `rgb_000012.jpg` pairs with `thermal_000012.jpg`.
- **Images must already be undistorted.** The pipeline uses a pinhole camera
  with no distortion.
- **`intrinsics.txt`** is one line, `fx fy cx cy`, in pixels, at the
  resolution of the images in `rgb/`. The file in `data/` holds example
  values. **Replace them with your camera's values:**

  ```
  1948.88 1948.88 810.0 462.5
  ```

  The pipeline reads this file once, at the start of the first step, and keeps
  these values fixed for the whole run. If a scene was shot with a different
  camera, put its own `intrinsics.txt` inside that scene's folder.
- Optional: list scene names (one per line) in `data/val_scenes.txt` and
  `data/test_scenes.txt`. Every other scene goes to train.

## 3. Run

```bash
./run.sh
```

This runs every step in order. If it stops, run the same command again. Scenes
that are already finished are skipped.

## 4. Output in `output/`

```
output/
├── sfm/
│   ├── master_results.csv          <- per-scene status: EXCELLENT / GOOD / WARN / SFM_FAIL / TOO_SMALL
│   └── <scene>/sparse/full/        <- COLMAP model: camera poses of the registered frames
├── mvs/
│   └── <scene>/mvs_refined_depth/  <- FINAL DEPTH MAPS, one <rgb image name>.h5 per registered frame
└── image_pairs/
    ├── pairs_npz/<scene>.npz       <- per scene: image paths, depth paths, intrinsics,
    │                                  cam2world poses, verified RGB-X pairs + overlap
    └── scene_info/{train,val,test}/<scene>.npy   <- FINAL DATASET (MegaDepth format)
```

- Depth `.h5` files store the depth map under the key `depth` (0 = no depth).
  Depth is in SfM units, not metres.
- A scene reaches the final dataset only if it is marked `EXCELLENT` or
  `GOOD` in `master_results.csv` and at least 100 of its pairs pass the
  depth-verified overlap check (0.30–0.92).
- Frames that SfM could not register get no pose and no depth, and are left out.

## Acknowledgements

The depth refinement (geometric/photometric depth fusion, stability filtering
and semantic masking) follows the post-processing recipe of MegaDepth.
