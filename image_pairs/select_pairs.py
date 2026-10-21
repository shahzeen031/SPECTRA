# Build the FINAL training NPZ set:
#   - RGB-X pairs (cross-modal task)
#   - depth_paths -> MVS-only refined depths (mvs_refined_depth/, no DA fill)
#   - pair lists replaced by depth-verified true-overlap pairs
#     (compute_depth_overlap.py output), band-filtered:
#         OV_MIN <= overlap <= OV_MAX  and  n_cons >= MIN_CONS
#     OV_MAX rejects near-duplicate frames, OV_MIN rejects weak co-visibility,
#     MIN_CONS guarantees enough verified pixels to sample correspondences
#     (n_cons is at 1/4 resolution; x16 for full-res pixel count).
#   - scenes with < MIN_PAIRS surviving pairs are dropped (reported).
#
# Output: $PAIRS_OUTPUT_ROOT/pairs_npz/{scene}.npz
#         + build_summary.json
# NPZ format identical to scene_npz;
# pair_overlaps now holds the depth-verified overlap.
# (Distinct from build_scene_npz.py, the original sparse-reprojection builder.)

import glob
import json
import os

import numpy as np

PAIRS_OUTPUT_ROOT = os.environ.get("PAIRS_OUTPUT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "image_pairs"))
SRC_NPZ = os.path.join(PAIRS_OUTPUT_ROOT, 'scene_npz')
OVERLAP_DIR = os.path.join(PAIRS_OUTPUT_ROOT, 'depth_overlap')
DST = os.path.join(PAIRS_OUTPUT_ROOT, 'pairs_npz')

OV_MIN = 0.30
OV_MAX = 0.92
MIN_CONS = 1000      # at 1/4 res  (~16k full-res verified co-visible pixels)
MIN_PAIRS = 100      # per scene


def main():
    os.makedirs(DST, exist_ok=True)
    summary = dict(config=dict(OV_MIN=OV_MIN, OV_MAX=OV_MAX, MIN_CONS=MIN_CONS,
                               MIN_PAIRS=MIN_PAIRS, overlap_dir=OVERLAP_DIR),
                   scenes={}, dropped=[])

    for src in sorted(glob.glob(os.path.join(SRC_NPZ, '*.npz'))):
        scene = os.path.splitext(os.path.basename(src))[0]
        ov_path = os.path.join(OVERLAP_DIR, f'{scene}.npz')
        if not os.path.exists(ov_path):
            print(f'{scene}: NO OVERLAP FILE — skipped')
            summary['dropped'].append(dict(scene=scene, reason='no overlap file'))
            continue

        d = dict(np.load(src, allow_pickle=True))
        ov = np.load(ov_path)
        sel = ((ov['overlap'] >= OV_MIN) & (ov['overlap'] <= OV_MAX) &
               (ov['n_cons'] >= MIN_CONS))
        n_sel = int(sel.sum())
        if n_sel < MIN_PAIRS:
            print(f'{scene}: only {n_sel} pairs survive — DROPPED')
            summary['dropped'].append(dict(scene=scene, reason=f'{n_sel} pairs < {MIN_PAIRS}'))
            continue

        d['depth_paths'] = np.char.replace(d['depth_paths'].astype(str),
                                           '/fused_depth/', '/mvs_refined_depth/')
        d['pair_idx0'] = ov['idx0'][sel].astype(np.int64)
        d['pair_idx1'] = ov['idx1'][sel].astype(np.int64)
        d['pair_overlaps'] = ov['overlap'][sel].astype(np.float32)

        np.savez(os.path.join(DST, f'{scene}.npz'), **d)
        gaps = d['pair_idx1'] - d['pair_idx0']
        summary['scenes'][scene] = dict(
            n_pairs=n_sel,
            n_candidates=int(len(ov['overlap'])),
            n_images=int(len(d['rgb_paths'])),
            usable_frames=int(ov['n_frames_usable']),
            gap_median=int(np.median(gaps)),
            overlap_median=float(np.median(d['pair_overlaps'])),
        )
        print(f'{scene}: {n_sel} pairs (of {len(ov["overlap"])} candidates), '
              f'median gap {int(np.median(gaps))}, median overlap '
              f'{np.median(d["pair_overlaps"]):.2f}')

    summary['total_pairs'] = int(sum(s['n_pairs'] for s in summary['scenes'].values()))
    summary['n_scenes'] = len(summary['scenes'])
    with open(os.path.join(DST, 'build_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nFINAL: {summary['n_scenes']} scenes, {summary['total_pairs']} pairs, "
          f"{len(summary['dropped'])} dropped -> {DST}")


if __name__ == '__main__':
    main()
