"""Semantic-distortion metric: 5-point landmark NME between clean and protected images.

Quantifies how much a protection method moves facial *structure* (eyes/nose/mouth),
independent of the identity-embedding attack. A bounded-eps additive perturbation (ours)
cannot move landmarks, so its NME ~ 0; a GAN that regenerates the image (NullSwap) shifts
facial geometry, so its NME is clearly larger.

NME (Normalized Mean Error) per image:
    NME = mean_k || L_clean[k] - L_adv[k] ||_2  /  d_iod
where d_iod = || L_clean[0] - L_clean[1] || (inter-ocular distance; RetinaFace points
0,1 are the eyes). Reported as percentage (×100).

Reuses the RetinaFace 5-point detector already wired into the attack pipeline
(attack.face_detection). Run from repo root:

    python contrastive_attack/semantic_distortion.py \
        --clean_dir contrastive_attack/valid_set_paired \
        --adv_dir   contrastive_attack/out_ssim_ti_5p \
        --out_csv   nme_ours.csv
"""
try:
    import ml_dtypes as _mld
    for _m in ('float4_e2m1fn', 'float6_e2m3fn', 'float6_e3m2fn', 'float8_e8m0fnu'):
        if not hasattr(_mld, _m):
            setattr(_mld, _m, _mld.float8_e4m3fn)
except Exception:
    pass

import argparse
import csv
import os
import sys

import cv2 as cv
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_RF = os.path.join(_HERE, 'RetinaFace_Pytorch')
if _RF not in sys.path:
    sys.path.insert(0, _RF)

from attack import face_detection  # noqa: E402


def load_retinaface(device):
    import torchvision_model as tvm
    rf = tvm.create_retinaface({'layer2': 1, 'layer3': 2, 'layer4': 3})
    sd = torch.load(os.path.join(_RF, 'model.pt'), map_location='cpu')
    pruned = {k[7:]: v for k, v in sd.items() if k[7:] in rf.state_dict()}
    rf.load_state_dict(pruned)
    rf = rf.to(device).eval()
    for p in rf.parameters():
        p.requires_grad_(False)
    return rf


def load_batch(path, files, img_size, device):
    arr = np.zeros((len(files), img_size, img_size, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        im = cv.imread(os.path.join(path, f))
        if im.shape[:2] != (img_size, img_size):
            im = cv.resize(im, (img_size, img_size))
        arr[i] = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(device)  # [B,3,H,W] [0,255] RGB
    return (t / 255. - 0.5) / 0.5  # [-1,1]


def main(args):
    device = torch.device(args.device)
    rf = load_retinaface(device)
    img_size = args.img_size

    clean_files = sorted(f for f in os.listdir(args.clean_dir) if f.endswith('.jpg'))
    adv_set = set(f for f in os.listdir(args.adv_dir) if f.endswith('.jpg'))
    files = [f for f in clean_files if f in adv_set]

    rows = []
    nmes = []
    n_fallback = 0
    # Canonical fallback IOD that face_detection emits on detection failure: |0.7W - 0.3W| = 0.4W.
    fallback_iod = 0.4 * img_size

    for i in range(0, len(files), args.batch_size):
        chunk = files[i:i + args.batch_size]
        clean = load_batch(args.clean_dir, chunk, img_size, device)
        adv = load_batch(args.adv_dir, chunk, img_size, device)
        with torch.no_grad():
            _, lc = face_detection(clean, rf, device)  # [B,5,2]
            _, la = face_detection(adv, rf, device)
        lc = lc.float().cpu().numpy()
        la = la.float().cpu().numpy()
        for b in range(len(chunk)):
            iod = float(np.linalg.norm(lc[b, 0] - lc[b, 1]))
            # Skip images where the clean face wasn't detected (fallback grid → degenerate NME).
            if abs(iod - fallback_iod) < 1e-3 or iod < 1e-3:
                n_fallback += 1
                continue
            per_pt = np.linalg.norm(lc[b] - la[b], axis=1)  # [5]
            nme = float(per_pt.mean() / iod) * 100.0  # percent
            nmes.append(nme)
            rows.append([chunk[b], f'{nme:.4f}', f'{iod:.2f}'])

    nmes = np.array(nmes, dtype=np.float64)
    print('\n=== semantic-distortion (5-point landmark NME) ===')
    print(f'clean_dir = {args.clean_dir}')
    print(f'adv_dir   = {args.adv_dir}')
    print(f'n_valid={len(nmes)}  n_skipped_fallback={n_fallback}')
    print(f'NME (%)  mean={nmes.mean():.4f}  std={nmes.std():.4f}  '
          f'median={np.median(nmes):.4f}  p90={np.percentile(nmes, 90):.4f}')

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['filename', 'nme_percent', 'iod_px'])
            w.writerows(rows)
        print(f'per-sample CSV -> {args.out_csv}')


parser = argparse.ArgumentParser()
parser.add_argument('--clean_dir', type=str, required=True)
parser.add_argument('--adv_dir', type=str, required=True)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--out_csv', type=str, default=None)
args = parser.parse_args()
main(args)
