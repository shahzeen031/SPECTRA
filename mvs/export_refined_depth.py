# Build the MVS-only (MegaDepth-refine) depth set from the fused depths.
#
# Depth fusion (depth_fusion.py) follows MegaDepth's recipe (geo/photo fusion,
# stability filter, semantic masking) and tags each pixel with a source
# channel. This step keeps only MVS-derived pixels (source==2).
#
# Input : output/{scene}/fused_depth/{frame}.h5   (depth f32, source u8)
# Output: output/{scene}/mvs_refined_depth/{frame}.h5  (same format/convention:
#         depth=0 -> invalid-or-sky, source: 2=MVS-derived, 0=invalid)

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np

OUT_ROOT = os.environ.get("MVS_OUTPUT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))


def convert_one(src_path):
    scene_dir = os.path.dirname(os.path.dirname(src_path))
    dst_dir = os.path.join(scene_dir, 'mvs_refined_depth')
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if os.path.exists(dst_path):
        return 'skip'
    try:
        with h5py.File(src_path, 'r') as h:
            depth = h['depth'][:]
            source = h['source'][:]
        keep = source == 2
        depth_out = np.where(keep, depth, np.float32(0.0)).astype(np.float32)
        source_out = np.where(keep, np.uint8(2), np.uint8(0))
        tmp_path = dst_path + '.tmp'
        with h5py.File(tmp_path, 'w') as h:
            h.create_dataset('depth', data=depth_out, compression='gzip', compression_opts=4)
            h.create_dataset('source', data=source_out, compression='gzip', compression_opts=4)
        os.replace(tmp_path, dst_path)
        return 'ok'
    except Exception as e:
        return f'FAIL {src_path}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(OUT_ROOT, '*', 'fused_depth', '*.h5')))
    print(f'{len(files)} fused h5 files found', flush=True)
    for scene_dir in sorted(glob.glob(os.path.join(OUT_ROOT, '*', 'fused_depth'))):
        os.makedirs(os.path.join(os.path.dirname(scene_dir), 'mvs_refined_depth'), exist_ok=True)

    n_ok = n_skip = n_fail = 0
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(convert_one, files, chunksize=32)):
            if res == 'ok':
                n_ok += 1
            elif res == 'skip':
                n_skip += 1
            else:
                n_fail += 1
                print(res, flush=True)
            if (i + 1) % 2000 == 0:
                print(f'{i + 1}/{len(files)} done', flush=True)

    print(f'DONE: ok={n_ok} skip={n_skip} fail={n_fail}', flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
