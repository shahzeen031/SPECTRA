# Depth-verified pair overlap for the final dataset.
#
# The scene_npz pair lists come from sparse-point reprojection overlap, which
# badly overstates co-visibility for wide-baseline pairs (pairs with stated
# overlap ~0.2 can share no content). This recomputes overlap from the MVS-only refined depths with a mutual
# depth-consistency check (MegaDepth-style co-visibility, occlusion-aware):
#
#   for pair (i, j): backproject frame i's valid depth to world points,
#   project into frame j; a pixel counts as co-visible iff it lands in-bounds
#   with positive depth AND |z_proj - depth_j| < REL_TOL * depth_j.
#   overlap(i,j) = min(frac_i->j, frac_j->i)   computed at 1/4 resolution.
#
# Candidates: all frame pairs with 0 < gap <= MAX_GAP (beyond ~50 frames the
# GT was shown to collapse from pose drift; sparse-overlap "loop" pairs are
# excluded deliberately).
#
# Output per scene: {OUT_DIR}/{scene}.npz with
#   idx0, idx1  (int32)  — npz-image indices, idx0 < idx1
#   overlap     (float32)— symmetric depth-verified overlap in [0,1]
#   n_cons      (int32)  — min directional count of consistent pixels (1/4 res)

import argparse
import os
import traceback
from multiprocessing import Pool

import h5py
import numpy as np

PAIRS_OUTPUT_ROOT = os.environ.get("PAIRS_OUTPUT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "image_pairs"))
NPZ_ROOT = os.path.join(PAIRS_OUTPUT_ROOT, 'scene_npz')  # per-scene RGB-X npz (poses/intrinsics/paths)
OUT_DIR = os.path.join(PAIRS_OUTPUT_ROOT, 'depth_overlap')
STRIDE = 4
MAX_GAP = 60
REL_TOL = 0.10
MIN_VALID_FRAC = 0.02   # skip frames with nearly-empty MVS depth


def load_scene(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    depth_paths = np.char.replace(d['depth_paths'].astype(str),
                                  '/fused_depth/', '/mvs_refined_depth/')
    return depth_paths, d['intrinsics'].astype(np.float64), d['cam2world'].astype(np.float64)


def frame_world_points(depth_path, K, c2w):
    """Valid pixels of one frame as world-space points (1/4 res). Returns
    (world_pts Nx3, n_valid, depth_lowres, K_lowres) or None if depth unusable."""
    with h5py.File(depth_path, 'r') as h:
        depth = h['depth'][::STRIDE, ::STRIDE].astype(np.float64)
    H, W = depth.shape
    valid = depth > 0
    n_valid = int(valid.sum())
    if n_valid < MIN_VALID_FRAC * H * W:
        return None
    S = np.diag([1.0 / STRIDE, 1.0 / STRIDE, 1.0])
    Kl = S @ K
    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u - Kl[0, 2]) / Kl[0, 0] * z
    y = (v - Kl[1, 2]) / Kl[1, 1] * z
    pts_cam = np.stack([x, y, z], 1)
    world = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world.astype(np.float32), n_valid, depth.astype(np.float32), Kl


def directional(world_pts, depth_j, Kl_j, w2c_j):
    """Count of frame-i points consistently visible in frame j."""
    H, W = depth_j.shape
    cam = world_pts.astype(np.float64) @ w2c_j[:3, :3].T + w2c_j[:3, 3]
    z = cam[:, 2]
    front = z > 1e-6
    u = np.round(cam[front, 0] / z[front] * Kl_j[0, 0] + Kl_j[0, 2]).astype(np.int64)
    v = np.round(cam[front, 1] / z[front] * Kl_j[1, 1] + Kl_j[1, 2]).astype(np.int64)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zf = u[inb], v[inb], z[front][inb]
    dj = depth_j[v, u]
    cons = (dj > 0) & (np.abs(zf - dj) < REL_TOL * dj)
    return int(cons.sum())


def process_scene(scene_name):
    try:
        out_path = os.path.join(OUT_DIR, f'{scene_name}.npz')
        if os.path.exists(out_path):
            return f'{scene_name}: skip (exists)'
        depth_paths, Ks, c2ws = load_scene(os.path.join(NPZ_ROOT, f'{scene_name}.npz'))
        N = len(depth_paths)

        frames = []
        for i in range(N):
            try:
                frames.append(frame_world_points(depth_paths[i], Ks[i], c2ws[i]))
            except OSError:
                frames.append(None)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]

        idx0, idx1, ovl, ncons = [], [], [], []
        for i in range(N):
            if frames[i] is None:
                continue
            wp_i, nv_i, d_i, Kl_i = frames[i]
            for j in range(i + 1, min(i + MAX_GAP + 1, N)):
                if frames[j] is None:
                    continue
                wp_j, nv_j, d_j, Kl_j = frames[j]
                c_ij = directional(wp_i, d_j, Kl_j, w2cs[j])
                if c_ij == 0:
                    continue
                c_ji = directional(wp_j, d_i, Kl_i, w2cs[i])
                ov = min(c_ij / nv_i, c_ji / nv_j)
                if ov <= 0:
                    continue
                idx0.append(i); idx1.append(j)
                ovl.append(ov); ncons.append(min(c_ij, c_ji))

        np.savez(out_path,
                 idx0=np.array(idx0, np.int32), idx1=np.array(idx1, np.int32),
                 overlap=np.array(ovl, np.float32), n_cons=np.array(ncons, np.int32),
                 n_frames=np.int32(N),
                 n_frames_usable=np.int32(sum(f is not None for f in frames)))
        return f'{scene_name}: {len(idx0)} pairs, {sum(f is not None for f in frames)}/{N} usable frames'
    except Exception:
        return f'{scene_name}: FAIL\n{traceback.format_exc()}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    import glob
    scenes = sorted(os.path.splitext(os.path.basename(f))[0]
                    for f in glob.glob(os.path.join(NPZ_ROOT, '*.npz')))
    print(f'{len(scenes)} scenes', flush=True)
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(process_scene, scenes):
            print(res, flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
