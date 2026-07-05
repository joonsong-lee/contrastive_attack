"""Run NullSwap generator on a directory of clean source images."""
try:
    import ml_dtypes as _mld
    for _m in ('float4_e2m1fn', 'float6_e2m3fn', 'float6_e3m2fn', 'float8_e8m0fnu'):
        if not hasattr(_mld, _m):
            setattr(_mld, _m, _mld.float8_e4m3fn)
except Exception:
    pass

import argparse
import os
import sys

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NullSwapGenerator


def main(args):
    device = torch.device(args.device)
    # Seed RNG so PerturbationBlock's torch.randn_like noise is deterministic across runs.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    G = NullSwapGenerator().to(device).eval()
    sd = torch.load(args.ckpt, map_location='cpu')
    G.load_state_dict(sd['G'] if isinstance(sd, dict) and 'G' in sd else sd)
    for p in G.parameters():
        p.requires_grad_(False)

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.jpg'))

    def batched(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i + bs]

    with torch.no_grad():
        for chunk in tqdm(list(batched(files, args.batch_size))):
            imgs = []
            for fn in chunk:
                im = cv.imread(os.path.join(args.input_dir, fn))
                if im is None:
                    imgs.append(None)
                    continue
                if im.shape[:2] != (256, 256):
                    im = cv.resize(im, (256, 256))
                imgs.append(cv.cvtColor(im, cv.COLOR_BGR2RGB))
            valid = [(i, im) for i, im in enumerate(imgs) if im is not None]
            if not valid:
                continue
            idxs, ims = zip(*valid)
            t = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to(device)  # [B,3,256,256] [0,255]
            t = t / 127.5 - 1.0                                                          # → [-1, 1]
            out = ((G(t) + 1.0) * 127.5).clamp(0, 255)                                   # [-1,1] → [0,255]
            out_np = out.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            for k, i in enumerate(idxs):
                bgr = cv.cvtColor(out_np[k], cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(args.output_dir, chunk[i]), bgr)


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
main(args)
