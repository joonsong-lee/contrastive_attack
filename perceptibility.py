"""Per-sample perceptibility metrics between clean and adversarial images.

Metrics: LPIPS (VGG), SSIM, PSNR, L_inf (max abs diff in [0,255]), L2 (mean per-pixel).
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

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def main(args):
    device = torch.device(args.device)
    lpips_model = lpips.LPIPS(net='vgg').to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    clean_files = sorted(f for f in os.listdir(args.clean_dir) if f.endswith('.jpg'))
    adv_files = set(f for f in os.listdir(args.adv_dir) if f.endswith('.jpg'))
    pairs = [f for f in clean_files if f in adv_files]

    rows = []
    lpips_vals, ssim_vals, psnr_vals, linf_vals, l2_vals = [], [], [], [], []

    for f in tqdm(pairs):
        c = cv.imread(os.path.join(args.clean_dir, f))
        a = cv.imread(os.path.join(args.adv_dir, f))
        if c is None or a is None:
            continue
        if c.shape != a.shape:
            a = cv.resize(a, (c.shape[1], c.shape[0]))
        c_rgb = cv.cvtColor(c, cv.COLOR_BGR2RGB)
        a_rgb = cv.cvtColor(a, cv.COLOR_BGR2RGB)

        diff = c_rgb.astype(np.int32) - a_rgb.astype(np.int32)
        linf = float(np.abs(diff).max())
        l2 = float(np.sqrt((diff ** 2).mean()))

        ssim_v = float(ssim_fn(c_rgb, a_rgb, channel_axis=2, data_range=255))
        psnr_v = float(psnr_fn(c_rgb, a_rgb, data_range=255)) if linf > 0 else float('inf')

        c_t = torch.from_numpy(c_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0
        a_t = torch.from_numpy(a_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0
        with torch.no_grad():
            lp_v = float(lpips_model(c_t, a_t).item())

        lpips_vals.append(lp_v)
        ssim_vals.append(ssim_v)
        if psnr_v != float('inf'):
            psnr_vals.append(psnr_v)
        linf_vals.append(linf)
        l2_vals.append(l2)
        rows.append([f, f'{lp_v:.6f}', f'{ssim_v:.6f}', f'{psnr_v:.4f}', f'{linf:.1f}', f'{l2:.4f}'])

    def stat(vs):
        a = np.array(vs, dtype=np.float64)
        return a.mean(), a.std()

    lp_m, lp_s = stat(lpips_vals)
    ss_m, ss_s = stat(ssim_vals)
    pn_m, pn_s = stat(psnr_vals) if psnr_vals else (float('inf'), 0.0)
    li_m, li_s = stat(linf_vals)
    l2_m, l2_s = stat(l2_vals)

    print('\n=== perceptibility summary (n={}) ==='.format(len(rows)))
    print(f'clean_dir = {args.clean_dir}')
    print(f'adv_dir   = {args.adv_dir}')
    print(f'LPIPS(VGG)  mean={lp_m:.4f} std={lp_s:.4f}')
    print(f'SSIM        mean={ss_m:.4f} std={ss_s:.4f}')
    print(f'PSNR (dB)   mean={pn_m:.2f}  std={pn_s:.2f}')
    print(f'L_inf (px)  mean={li_m:.2f}  std={li_s:.2f}')
    print(f'L2 (RMSE)   mean={l2_m:.4f} std={l2_s:.4f}')

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['filename', 'lpips_vgg', 'ssim', 'psnr', 'linf', 'l2_rmse'])
            w.writerows(rows)
        print(f'per-sample CSV -> {args.out_csv}')


parser = argparse.ArgumentParser()
parser.add_argument('--clean_dir', type=str, required=True)
parser.add_argument('--adv_dir', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--out_csv', type=str, default=None)
args = parser.parse_args()
main(args)
