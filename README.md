# RGB-X depth and pair dataset pipeline

Turns image sequences from an RGB camera plus a pixel-aligned second modality
(X: thermal, SWIR, NIR, ...) into camera poses, depth maps and verified
RGB↔X training pairs. The pipeline computes poses and depth from the **RGB images only**.
The X images reuse the poses and depth of their matching RGB frames.

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
