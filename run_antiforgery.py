"""Run AntiForgery (Wang et al. IJCAI 2022) on a directory of 256x256 jpgs.

Wraps `lab_attack` from the official AntiForgery repo to produce perturbed source
images in our standard k.jpg numbered format. Source set independent (CelebA-HQ
or LFW); reads `<input_dir>/N.jpg`, writes `<output_dir>/N.jpg`.
"""
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

AF_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'AntiForgery')
sys.path.insert(0, AF_ROOT)
from model import Generator
from utils import create_labels, denorm
from color_space import rgb2lab, lab2rgb
import torch.nn as nn
from torchvision import transforms as T


def lab_attack_safe(X_nat, c_trg, model, epsilon=0.05, iter=100):
    """NaN-guarded version of AntiForgery's lab_attack. Adds:
    - X_new clamped to [-1, 1] before model forward (lab2rgb can overflow).
    - Per-iter pert_a.grad NaN replacement and grad-norm clipping.
    - Final x_adv NaN replacement.
    Algorithm and hyperparameters otherwise unchanged from the original."""
    criterion = nn.MSELoss().cuda()
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))
    X = denorm(X_nat.clone())  # [0, 1]
    norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    for i in range(iter):
        X_lab = rgb2lab(X).cuda()
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        # avoid in-place to keep autograd happy
        X_lab = torch.cat([X_lab[:, :1], X_lab[:, 1:] + pert], dim=1)
        X_new = norm(lab2rgb(X_lab)).clamp(-1.0, 1.0)

        with torch.no_grad():
            gen_noattack, _ = model(X_nat, c_trg[i % len(c_trg)])
        gen_stargan, _ = model(X_new, c_trg[i % len(c_trg)])
        loss = -criterion(gen_stargan, gen_noattack)

        optimizer.zero_grad()
        loss.backward()
        if pert_a.grad is not None:
            torch.nan_to_num_(pert_a.grad, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nn.utils.clip_grad_norm_([pert_a], max_norm=1.0)
        optimizer.step()

    # Final perturbed image, NaN-cleaned and clamped.
    with torch.no_grad():
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab = rgb2lab(X).cuda()
        X_lab = torch.cat([X_lab[:, :1], X_lab[:, 1:] + pert], dim=1)
        X_new = norm(lab2rgb(X_lab))
        X_new = torch.nan_to_num(X_new, nan=0.0).clamp(-1.0, 1.0)
    return X_new, X_new - X_nat


def load_imgs(input_dir):
    def numeric_key(f):
        try:
            return int(f.split('.')[0])
        except ValueError:
            return -1
    files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') and f != 'target.jpg']
    return sorted(files, key=numeric_key)


def main(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # Load StarGAN G with 5-attribute config (matches their pretrained 256x256).
    G = Generator(conv_dim=64, c_dim=5, repeat_num=6).to(device)
    sd = torch.load(args.ckpt, map_location='cpu')
    G.load_state_dict(sd)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    files = load_imgs(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

    for i in tqdm(range(0, len(files), args.batch_size)):
        batch_files = files[i:i + args.batch_size]
        ims = []
        for f in batch_files:
            im = cv.imread(os.path.join(args.input_dir, f))
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            if im.shape[:2] != (256, 256):
                im = cv.resize(im, (256, 256))
            ims.append(im)
        x = np.stack(ims).astype(np.float32) / 127.5 - 1.0  # [B,H,W,3] in [-1,1]
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)  # [B,3,256,256]

        # Build c_trg list — 5 attribute targets, all-zero c_org for unknown source labels.
        c_org = torch.zeros(x.shape[0], 5, device=device)
        c_trg_list = create_labels(c_org, c_dim=5, dataset='CelebA',
                                    selected_attrs=selected_attrs)

        with torch.enable_grad():
            x_adv, _ = lab_attack_safe(x, c_trg_list, G,
                                        epsilon=args.epsilon, iter=args.iter)

        # x_adv is in [-1, 1] after T.Normalize (mean=0.5, std=0.5) — same scale.
        out_255 = ((x_adv + 1.0) * 127.5).clamp(0, 255)
        out_np = out_255.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        for k, f in enumerate(batch_files):
            bgr = cv.cvtColor(out_np[k], cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(args.output_dir, f), bgr)


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--ckpt', type=str,
                    default='../AntiForgery/stargan_celeba_256/models/200000-G.ckpt')
parser.add_argument('--iter', type=int, default=100)
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
main(args)
