"""Build one RGB->X training-pair NPZ per scene (X images live in {scene}/x/,
pixel-aligned with rgb/).

Pipeline:
  1. Load the scene's sparse/{full or best} COLMAP model (poses + sparse
     points) — same model the fusion stage used for that scene.
  2. Keep only images that (a) have an RGB file on disk and (b) have a
     completed fused-depth h5 from the MVS/fusion stage.
  3. Overlap between every pair of kept images = simple 3D-point
     reprojection: project image i's own SfM points into every other
     camera, check they land in-frame and in front of it. Symmetric score
     = min(i->j, j->i). No dense/fused depth involved in this step.
  4. Candidate pairs (overlap >= MIN_OVERLAP) are emitted once per
     direction that has a X frame available: (i, j) with i=RGB anchor,
     j=X target, for every direction where the X-side frame exists.
"""
import re
from pathlib import Path

import numpy as np
import pycolmap

import pairs_config as cfg

_FRAME_RE = re.compile(r"(\d+)")


def frame_index(name: str) -> int:
    m = _FRAME_RE.findall(name)
    return int(m[-1]) if m else 0


def build_x_lookup(x_dir: Path) -> dict[int, Path]:
    """frame number -> x path, matched by trailing frame number regardless
    of filename prefix (varies by scene: x_original_* / x_image_* /
    others — verified NOT to be a fixed convention across the dataset)."""
    out = {}
    if not x_dir.exists():
        return out
    for p in x_dir.iterdir():
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = _FRAME_RE.findall(p.stem)
        if m:
            out[int(m[-1])] = p
    return out


def _pose_w2c(image):
    p = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    R = np.asarray(p.rotation.matrix() if callable(p.rotation.matrix)
                   else p.rotation.matrix, dtype=np.float64)
    t = np.asarray(p.translation, dtype=np.float64)
    return R, t


def load_model(scene: str, model_used: str) -> pycolmap.Reconstruction:
    tag = "best" if model_used == "best_fallback" else "full"
    return pycolmap.Reconstruction(cfg.SFM_OUTPUT_ROOT / scene / "sparse" / tag)


def build_pose_and_points(rec: pycolmap.Reconstruction):
    """All registered images, sorted by frame index.

    Returns names, K(3,3), cam2world(N,4,4), world2cam R/t arrays, and a
    per-image list of its own SfM points' world xyz (Ni,3) for reprojection.
    """
    img_ids = sorted(rec.reg_image_ids(),
                     key=lambda i: frame_index(rec.images[i].name))
    names = [rec.images[i].name for i in img_ids]
    cam = next(iter(rec.cameras.values()))
    fx, fy, cx, cy = [float(x) for x in cam.params]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    W, H = cam.width, cam.height

    N = len(img_ids)
    R_w2c = np.zeros((N, 3, 3))
    t_w2c = np.zeros((N, 3))
    cam2world = np.zeros((N, 4, 4), dtype=np.float32)
    points_per_image = []

    for k, iid in enumerate(img_ids):
        im = rec.images[iid]
        R, t = _pose_w2c(im)
        R_w2c[k], t_w2c[k] = R, t
        M = np.eye(4); M[:3, :3] = R.T; M[:3, 3] = -R.T @ t
        cam2world[k] = M.astype(np.float32)

        pids = [p.point3D_id for p in im.points2D if p.has_point3D()]
        pts = (np.array([rec.points3D[pid].xyz for pid in pids], dtype=np.float64)
              if pids else np.zeros((0, 3)))
        points_per_image.append(pts)

    return names, K, W, H, cam2world, R_w2c, t_w2c, points_per_image


def reprojection_overlap_matrix(R_w2c, t_w2c, points_per_image,
                                fx, fy, cx, cy, W, H) -> np.ndarray:
    """Symmetric (N,N) overlap matrix via sparse-point reprojection.

    overlap_from[i, j] = fraction of image i's own SfM points that project
    validly (in front of, and inside the frame of) camera j. Vectorized
    over j and over the points of each i in turn.
    """
    N = len(points_per_image)
    overlap_from = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        P = points_per_image[i]
        if P.shape[0] == 0:
            continue
        # (N,3,3) x (P,3) -> (N,P,3): project P's points into every camera
        cam_pts = np.einsum('nij,pj->npi', R_w2c, P) + t_w2c[:, None, :]
        z = cam_pts[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * cam_pts[..., 0] / z + cx
            v = fy * cam_pts[..., 1] / z + cy
        valid = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        overlap_from[i] = valid.mean(axis=1)
    np.fill_diagonal(overlap_from, 0.0)
    return np.minimum(overlap_from, overlap_from.T)


def build_scene_npz(scene: str, model_used: str, image_dir: Path,
                    mvs_workspace: Path, min_overlap: float = cfg.MIN_OVERLAP) -> dict:
    rec = load_model(scene, model_used)
    names, K, W, H, cam2world, R_w2c, t_w2c, points_per_image = \
        build_pose_and_points(rec)

    x_dir = image_dir.parent / "x"
    depth_dir = mvs_workspace / "fused_depth"
    x_lookup = build_x_lookup(x_dir)

    keep_idx, rgb_paths, x_paths, depth_paths = [], [], [], []
    x_present = []
    skipped_no_rgb, skipped_no_depth = 0, 0
    for k, name in enumerate(names):
        rgb_p = image_dir / name
        dep_p = depth_dir / f"{name}.h5"
        if not rgb_p.exists():
            skipped_no_rgb += 1
            continue
        if not dep_p.exists():
            skipped_no_depth += 1
            continue
        x_p = x_lookup.get(frame_index(name))
        keep_idx.append(k)
        rgb_paths.append(str(rgb_p))
        depth_paths.append(str(dep_p))
        x_paths.append(str(x_p) if x_p is not None else "")
        x_present.append(x_p is not None)

    keep_idx = np.array(keep_idx, dtype=np.int64)
    n_kept = len(keep_idx)
    K_out = np.tile(K.astype(np.float32)[None], (n_kept, 1, 1))
    cam2world_out = cam2world[keep_idx]
    R_w2c_k = R_w2c[keep_idx]
    t_w2c_k = t_w2c[keep_idx]
    points_k = [points_per_image[k] for k in keep_idx]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    overlap = reprojection_overlap_matrix(R_w2c_k, t_w2c_k, points_k,
                                          fx, fy, cx, cy, W, H)

    x_present = np.array(x_present)
    pair_idx0, pair_idx1, pair_overlaps = [], [], []
    iu, ju = np.triu_indices(n_kept, k=1)
    ov = overlap[iu, ju]
    cand = ov >= min_overlap
    for i, j, s in zip(iu[cand], ju[cand], ov[cand]):
        if x_present[j]:
            pair_idx0.append(i); pair_idx1.append(j); pair_overlaps.append(s)
        if x_present[i]:
            pair_idx0.append(j); pair_idx1.append(i); pair_overlaps.append(s)

    out_path = cfg.OUTPUT_DIR / f"{scene}.npz"
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rgb_paths=np.array(rgb_paths),
        x_paths=np.array(x_paths),
        depth_paths=np.array(depth_paths),
        intrinsics=K_out,
        cam2world=cam2world_out,
        pair_idx0=np.array(pair_idx0, dtype=np.int64),
        pair_idx1=np.array(pair_idx1, dtype=np.int64),
        pair_overlaps=np.array(pair_overlaps, dtype=np.float32),
    )

    n_pairs = len(pair_idx0)
    n_x_available = int(x_present.sum())
    return {
        "scene": scene, "model_used": model_used,
        "n_images_model": len(names), "n_images_kept": n_kept,
        "skipped_no_rgb": skipped_no_rgb, "skipped_no_depth": skipped_no_depth,
        "n_x_available": n_x_available,
        "n_pairs": n_pairs,
        "overlap_mean": float(ov[cand].mean()) if cand.any() else 0.0,
        "overlap_median": float(np.median(ov[cand])) if cand.any() else 0.0,
        "npz_path": str(out_path),
    }
