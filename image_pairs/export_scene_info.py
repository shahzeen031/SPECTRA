# Convert the final scene NPZs into MegaDepth-style scene_info .npy files
# (the format RoMa / DKM / LoFTR-style MegaDepth loaders read directly).
#
# scene_info contract (see romatch/datasets/megadepth.py):
#   image_paths (list[str]), depth_paths (list[str]), intrinsics (K,3,3),
#   poses (K,4,4 world2cam), pairs (M,2 int), overlaps (M,)
#
# RGB-X trick: images 0..N-1 are the RGB frames, N..2N-1 the X frames.
# RGB and X frames are pixel-aligned, so both halves share the same
# intrinsics, pose, and MVS depth.  Each verified pair (i, j) is emitted in
# two cross-modal directions:
#   (i, N+j)  -> im_A = RGB frame i,  im_B = X frame j
#   (N+j, i)  -> im_A = X frame j,    im_B = RGB frame i
#
# NPZ poses are cam2world -> inverted here (MegaDepth poses are world2cam).
#
# Split: scenes listed in {DATA_ROOT}/val_scenes.txt or test_scenes.txt (one
# scene name per line, optional) go to val/ or test/, all others to train/.

import glob
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS_OUTPUT_ROOT = os.environ.get("PAIRS_OUTPUT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "image_pairs"))
NPZ_ROOT = os.path.join(PAIRS_OUTPUT_ROOT, 'pairs_npz')
OUT_ROOT = os.path.join(PAIRS_OUTPUT_ROOT, 'scene_info')
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(_HERE), "data"))


def read_scene_list(name):
    p = os.path.join(DATA_ROOT, name)
    if not os.path.exists(p):
        return set()
    with open(p) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith('#')}


def convert(npz_path, out_path):
    d = np.load(npz_path, allow_pickle=True)
    N = len(d['rgb_paths'])

    image_paths = np.concatenate([d['rgb_paths'].astype(str),
                                  d['x_paths'].astype(str)])
    depth_paths = np.concatenate([d['depth_paths'].astype(str)] * 2)
    intrinsics = np.tile(d['intrinsics'].astype(np.float64), (2, 1, 1))
    w2c = np.linalg.inv(d['cam2world'].astype(np.float64))
    poses = np.tile(w2c, (2, 1, 1))

    i, j = d['pair_idx0'].astype(np.int64), d['pair_idx1'].astype(np.int64)
    pairs = np.concatenate([np.stack([i, N + j], 1),
                            np.stack([N + j, i], 1)])
    overlaps = np.concatenate([d['pair_overlaps']] * 2).astype(np.float64)

    scene_info = dict(image_paths=image_paths, depth_paths=depth_paths,
                      intrinsics=intrinsics, poses=poses,
                      pairs=pairs, overlaps=overlaps)
    np.save(out_path, scene_info, allow_pickle=True)
    return N, len(pairs)


def main():
    val_scenes = read_scene_list('val_scenes.txt')
    test_scenes = read_scene_list('test_scenes.txt')
    tot = {'train': 0, 'val': 0, 'test': 0}
    for npz_path in sorted(glob.glob(os.path.join(NPZ_ROOT, '*.npz'))):
        scene = os.path.splitext(os.path.basename(npz_path))[0]
        split = ('test' if scene in test_scenes else
                 'val' if scene in val_scenes else 'train')
        os.makedirs(os.path.join(OUT_ROOT, split), exist_ok=True)
        out_path = os.path.join(OUT_ROOT, split, f'{scene}.npy')
        n_img, n_pairs = convert(npz_path, out_path)
        tot[split] += n_pairs
        print(f'{scene:<30} [{split}] {n_img} frames -> {n_pairs} directed pairs')
    print(f"\ntrain pairs: {tot['train']}   val pairs: {tot['val']}   test pairs: {tot['test']}")
    print(f'scene_info written to {OUT_ROOT}')


if __name__ == '__main__':
    main()
