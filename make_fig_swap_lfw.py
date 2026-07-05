"""Qualitative LFW face-swap figures (paper Fig. 5 and Fig. 6).

Shows that swapping with the PROTECTED source fails to carry the source identity,
on the out-of-distribution LFW set. Two modes:
  --mode samples  : one swapper, several LFW samples (depth)  -> Fig. 5
  --mode swappers : one LFW sample, all three swappers (breadth) -> Fig. 6

Columns per row: [Source | Target | Swap w/ clean source | Swap w/ protected source].
The last two share the same target; only the source (clean vs protected) differs, so any
identity change is attributable to the protection.

Run from repo root, e.g.:
  python contrastive_attack/make_fig_swap_lfw.py --mode samples --swapper simswap \
      --target_idx 0 --ids 0,10,25,42 --out CVPR_2026_Submission_Template__1_/fig_lfw_swap.jpg
"""
import argparse
import os

import cv2 as cv
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(ROOT, 'valid_set_paired_lfw')
LFW = os.path.join(ROOT, 'result_mt_ours_lfw')
TARGETS = os.path.join(ROOT, '..', 'targets')
S = 256


def rgb(path):
    im = cv.imread(path)
    if im is None:
        return np.full((S, S, 3), 200, np.uint8)
    if im.shape[:2] != (S, S):
        im = cv.resize(im, (S, S))
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)


def labelbar(w, text, h=28):
    bar = np.full((h, w, 3), 255, np.uint8)
    cv.putText(bar, text, (6, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)
    return bar


def rowlabel(text, h=S, w=150):
    bar = np.full((h, w, 3), 255, np.uint8)
    cv.putText(bar, text, (6, h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv.LINE_AA)
    return bar


def swap_paths(swapper, tgt, idx):
    d = os.path.join(LFW, f'target_{tgt}')
    return (os.path.join(d, f'{swapper}_original', f'{idx}.jpg'),
            os.path.join(d, f'{swapper}_pertur', f'{idx}.jpg'))


def build_row(row_label, src_idx, tgt, swapper):
    src = rgb(os.path.join(CLEAN, f'{src_idx}.jpg'))
    tgt_im = rgb(os.path.join(TARGETS, f'target_{tgt}.jpg'))
    o, p = swap_paths(swapper, tgt, src_idx)
    cells = [src, tgt_im, rgb(o), rgb(p)]
    strip = np.concatenate(cells, axis=1)
    return np.concatenate([rowlabel(row_label), strip], axis=1)


def main(a):
    col_titles = ['Source', 'Target', 'Swap w/ clean', 'Swap w/ protected']
    header = np.concatenate([np.full((28, 150, 3), 255, np.uint8)]
                            + [labelbar(S, t) for t in col_titles], axis=1)
    rows = [header]
    if a.mode == 'samples':
        for idx in [int(x) for x in a.ids.split(',')]:
            rows.append(build_row(f'#{idx}', idx, a.target_idx, a.swapper))
    else:  # swappers
        idx = int(a.ids.split(',')[0])
        names = {'facedancer': 'FaceDancer', 'simswap': 'SimSwap', 'hifi': 'HifiFace'}
        for sw in ['facedancer', 'simswap', 'hifi']:
            rows.append(build_row(names[sw], idx, a.target_idx, sw))
    grid = np.concatenate(rows, axis=0)
    cv.imwrite(a.out, cv.cvtColor(grid, cv.COLOR_RGB2BGR),
               [cv.IMWRITE_JPEG_QUALITY, 92])
    print(f'[fig] wrote {a.out}  {grid.shape[1]}x{grid.shape[0]}')


ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['samples', 'swappers'], required=True)
ap.add_argument('--swapper', default='simswap')
ap.add_argument('--target_idx', type=int, default=0)
ap.add_argument('--ids', required=True, help='comma-separated LFW source indices')
ap.add_argument('--out', required=True)
a = ap.parse_args()
main(a)
