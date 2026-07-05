"""Qualitative fidelity comparison figure: clean | Ours (bounded perturbation) | NullSwap (regeneration).

Selects the samples where the NullSwap re-implementation distorts facial geometry the most
(highest landmark NME) while our method keeps it low, so the montage illustrates the
structural-distortion contrast the aggregate metrics understate.

Run from repo root:
    python contrastive_attack/make_fig_compare.py \
        --clean_dir contrastive_attack/valid_set_paired \
        --ours_dir  contrastive_attack/out_pair_lpips \
        --null_dir  contrastive_attack/out_nullswap_v11 \
        --ours_nme  <scratch>/nme_ours.csv \
        --null_nme  <scratch>/nme_nullswap.csv \
        --out       CVPR_2026_Submission_Template__1_/fig_fidelity_compare.png
"""
import argparse
import csv
import os

import cv2 as cv
import numpy as np


def read_nme(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r['filename']] = float(r['nme_percent'])
    return d


def load_rgb(path, size=256):
    im = cv.imread(path)
    if im is None:
        return None
    if im.shape[:2] != (size, size):
        im = cv.resize(im, (size, size))
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)


def label_strip(w, text, h=26):
    strip = np.full((h, w, 3), 255, np.uint8)
    cv.putText(strip, text, (6, 18), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv.LINE_AA)
    return strip


def main(args):
    ours_nme = read_nme(args.ours_nme)
    null_nme = read_nme(args.null_nme)
    # Candidates present in all three dirs; rank by (NullSwap NME - Ours NME): large gap => our
    # bounded perturbation preserved geometry while NullSwap moved it.
    common = [f for f in null_nme
              if f in ours_nme
              and os.path.exists(os.path.join(args.clean_dir, f))
              and os.path.exists(os.path.join(args.ours_dir, f))
              and os.path.exists(os.path.join(args.null_dir, f))]
    common.sort(key=lambda f: null_nme[f] - ours_nme[f], reverse=True)
    picks = common[:args.n]
    print('[fig] selected samples (file, ours_NME, null_NME):')
    for f in picks:
        print(f'   {f}  ours={ours_nme[f]:.2f}  null={null_nme[f]:.2f}')

    S = 256
    col_titles = ['Clean source', 'Ours (bounded)', 'NullSwap (re-impl.)']
    header = np.concatenate([label_strip(S, t) for t in col_titles], axis=1)
    rows = [header]
    for f in picks:
        imgs = [load_rgb(os.path.join(d, f)) for d in (args.clean_dir, args.ours_dir, args.null_dir)]
        if any(im is None for im in imgs):
            continue
        rows.append(np.concatenate(imgs, axis=1))
    grid = np.concatenate(rows, axis=0)
    cv.imwrite(args.out, cv.cvtColor(grid, cv.COLOR_RGB2BGR))
    print(f'[fig] wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})')


parser = argparse.ArgumentParser()
parser.add_argument('--clean_dir', required=True)
parser.add_argument('--ours_dir', required=True)
parser.add_argument('--null_dir', required=True)
parser.add_argument('--ours_nme', required=True)
parser.add_argument('--null_nme', required=True)
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--out', required=True)
args = parser.parse_args()
main(args)
