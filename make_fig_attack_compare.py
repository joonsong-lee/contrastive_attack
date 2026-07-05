"""Figure: naive antipodal attack vs. our Contrastive Attack (gradient-vanishing motivation).

Both attacks push the source embedding away from its own identity. The naive baseline targets the
exact antipodal vector -theta(x); by Eq. (8-9) the cosine gradient is proportional to
sin(angle(x, target)) ~ 0 at the start, so under plain gradient descent it stalls at the saddle and
cosine-to-source barely drops. Our Contrastive Attack adds attribute-variant negatives (source +
flip/RRC views), which sit at a non-zero angle and supply a real gradient, so cosine-to-source
collapses.

--regime vanilla : plain gradient descent -> illustrates the saddle (naive stalls, ours escapes).
--regime sign    : sign-gradient MI-FGSM -> the sign op amplifies the tiny gradient, so BOTH escape
                   (documents that the practical optimizer mitigates the saddle).

Run from the contrastive_attack/ directory:
    cd contrastive_attack && python make_fig_attack_compare.py --regime vanilla \
        --out ../CVPR_2026_Submission_Template__1_/fig_attack_compare.jpg
"""
import argparse
import os
import sys

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.append('./AdaFace/')
import AdaFace.net as adanet
sys.path.remove('./AdaFace/')
sys.path.append('./arcface_torch')
from arcface_torch.backbones.iresnet import iresnet50
sys.path.remove('./arcface_torch')

from attack import contrastive_opposite
from util import resize_to_recognition


def load_nets(device):
    resnet = iresnet50().to(device)
    resnet.load_state_dict(torch.load('./arcface_torch/backbone.pth'))
    resnet.eval()
    ada = adanet.build_model('ir_50')
    sd = torch.load('./AdaFace/adaface_ir50_webface4m.ckpt')['state_dict']
    ada.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')})
    ada = ada.eval().to(device)
    resnet2 = iresnet50().to(device)
    resnet2.load_state_dict(torch.load('./arcface_torch/gli_backbone.pth'))
    resnet2.eval()
    return nn.ModuleDict({"arc": resnet2, "arc2": resnet, "ada": ada})


def load_batch(src_dir, ids, device):
    ims = []
    for i in ids:
        im = cv.imread(os.path.join(src_dir, f'{i}.jpg'))
        if im.shape[:2] != (256, 256):
            im = cv.resize(im, (256, 256))
        ims.append(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    t = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to(device)  # [B,3,256,256] [0,255]
    return t


def run(attack, batch_o, mode, regime, iters, lr, eps):
    """mode: 'naive' | 'cont'.  regime: 'vanilla' (plain GD) | 'sign' (MI-FGSM).
    Returns [n_nets, iters] cosine-to-source trajectory."""
    keys = list(attack.nets.keys())
    batch = (batch_o / 255. - 0.5) / 0.5
    with torch.no_grad():
        ori = [attack._encode(resize_to_recognition(batch), k).detach() for k in keys]
        ort = [F.normalize(-o) for o in ori]
    g = torch.zeros_like(batch)
    delta = torch.zeros_like(batch, requires_grad=True)
    sims = np.zeros((len(keys), iters))
    for it in range(iters):
        b_t_small = resize_to_recognition(batch + delta)
        aug = attack._encode_multi_aug([resize_to_recognition(batch)]) if mode == 'cont' else None  # [4B,D]
        losses = []
        for j, k in enumerate(keys):
            feat = attack._encode(b_t_small, k)
            if mode == 'naive':
                losses.append(F.cosine_similarity(feat, ori[j], dim=1).mean())  # minimize -> antipodal
            else:
                target = torch.cat([ort[j], ori[j], aug[j]], dim=0)  # block0=ort(+), rest negatives
                losses.append(attack.cont_loss(feat, target))
            sims[j, it] = F.cosine_similarity(feat, ori[j], dim=1).mean().item()
        loss = torch.stack(losses).mean()
        loss.backward()
        grad = delta.grad.clone()
        delta.grad.zero_()
        if regime == 'sign':
            g = g + grad / torch.norm(grad, p=1, dim=(1, 2, 3), keepdim=True)
            delta.data = delta.data - lr * torch.sign(g)
        else:  # vanilla: plain gradient descent, no sign, no normalization
            delta.data = delta.data - lr * grad
        delta.data = delta.data.clamp(-eps / 127, eps / 127)
        delta.data = (batch + delta.data).clamp(-1, 1) - batch
    return sims


def main(a):
    device = torch.device(a.device)
    nets = load_nets(device)
    attack = contrastive_opposite(device, nets, lpips_weight=0.0, loss_type='ce', regularizer='none')
    batch_o = load_batch(a.src_dir, list(range(a.batch_size)), device)

    lr = a.lr if a.lr is not None else (0.001 if a.regime == 'sign' else 2.0)
    print(f'[fig] regime={a.regime} lr={lr}  running naive...')
    naive = run(attack, batch_o, 'naive', a.regime, a.iter, lr, a.eps)
    print('[fig] running contrastive (ours)...')
    cont = run(attack, batch_o, 'cont', a.regime, a.iter, lr, a.eps)

    nm, cm = naive.mean(0), cont.mean(0)
    x = np.arange(a.iter)
    plt.figure(figsize=(7, 4.3))
    plt.plot(x, nm, color='#d62728', lw=2.2, label='Naive antipodal attack')
    plt.plot(x, cm, color='#1f77b4', lw=2.2, label='Contrastive Attack (Ours)')
    plt.xlabel('Iteration'); plt.ylabel('Cosine similarity to source identity')
    plt.title('Naive antipodal vs. Contrastive Attack')
    plt.grid(True, ls='--', alpha=0.5); plt.legend(loc='best')
    plt.ylim(min(-0.1, cm.min() - 0.05), 1.02)
    plt.tight_layout()
    plt.savefig(a.out, dpi=150)
    print(f'[fig] wrote {a.out}  naive_final={nm[-1]:.3f} ours_final={cm[-1]:.3f}')


ap = argparse.ArgumentParser()
ap.add_argument('--src_dir', default='./valid_set_paired')
ap.add_argument('--batch_size', type=int, default=16)
ap.add_argument('--iter', type=int, default=300)
ap.add_argument('--eps', type=int, default=8)
ap.add_argument('--lr', type=float, default=None)
ap.add_argument('--regime', choices=['vanilla', 'sign'], default='vanilla')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
a = ap.parse_args()
main(a)
