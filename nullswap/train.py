"""Train NullSwap generator + PatchGAN discriminator on CelebA-HQ."""
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
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NullSwapGenerator, PatchDiscriminator
from loss import identity_cos_loss, DynamicLossWeighter, discriminator_loss, generator_gan_loss

# Face recognition backbones + RetinaFace alignment — reuse from the main attack pipeline.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'arcface_torch'))
from backbones.iresnet import iresnet50

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RetinaFace_Pytorch'))
import torchvision_model as rf_tvm
from attack import face_detection
from util import differentiable_warping

from facenet_pytorch import InceptionResnetV1
import lpips


def load_retinaface(device):
    rl = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    rf = rf_tvm.create_retinaface(rl)
    rf_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RetinaFace_Pytorch')
    sd = torch.load(os.path.join(rf_root, 'model.pt'), map_location='cpu')
    pruned = {k[7:]: v for k, v in sd.items() if k[7:] in rf.state_dict()}
    rf.load_state_dict(pruned)
    rf = rf.to(device).eval()
    for p in rf.parameters():
        p.requires_grad_(False)
    return rf


@torch.no_grad()
def aligned_crop(images_m11, retinaface, device, target_size):
    """Detect landmarks on [-1,1] RGB, return aligned crop in [-1,1]."""
    boxes, lnds = face_detection(images_m11, retinaface, device)
    warped, _ = differentiable_warping(device, images_m11, b_boxs=boxes, b_lnds=lnds,
                                       affine_arrays=None, mi_transform=False,
                                       target_shape=target_size)
    return warped


def arc_embed(arc, aligned_112_m11):
    return F.normalize(arc(aligned_112_m11), dim=1)


def facenet_embed(facenet, aligned_160_m11):
    return F.normalize(facenet(aligned_160_m11), dim=1)


def main(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    print('[nullswap] loading dataset...')
    ds = load_dataset('korexyz/celeba-hq-256x256', split=f'train[:{args.num_train}]',
                      cache_dir=args.data_root)
    def tfm(eg):
        c = Compose([PILToTensor()])
        eg['image'] = [(c(img).float() / 127.5 - 1.0) for img in eg['image']]
        return eg
    ds = ds.with_transform(tfm)
    def collate(examples):
        return torch.stack([e['image'] for e in examples])
    loader = DataLoader(ds, collate_fn=collate, batch_size=args.batch_size,
                        shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    print('[nullswap] loading backbones...')
    retinaface = load_retinaface(device)
    arcface_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'arcface_torch')
    arc = iresnet50().to(device)
    arc.load_state_dict(torch.load(os.path.join(arcface_root, 'backbone.pth'), map_location='cpu'))
    arc.eval()
    for p in arc.parameters():
        p.requires_grad_(False)
    facenet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    for p in facenet.parameters():
        p.requires_grad_(False)

    # Paper uses AlexNet backbone for LPIPS.
    lpips_loss = lpips.LPIPS(net='alex').to(device).eval()
    for p in lpips_loss.parameters():
        p.requires_grad_(False)

    G = NullSwapGenerator().to(device)
    D = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    scaler_g = torch.cuda.amp.GradScaler(enabled=args.amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=args.amp)

    dlw = DynamicLossWeighter(n_models=2)  # arcface + facenet
    os.makedirs(args.save_dir, exist_ok=True)

    LAMBDA_ID = 0.08
    LAMBDA_MSE = 1.8
    LAMBDA_LPIPS = 1.2
    LAMBDA_D = 0.1

    accum = max(1, args.accum_steps)
    effective_bs = args.batch_size * accum
    global_iter = 0
    effective_iter = 0
    t0 = time.time()
    accum_L_arc = 0.0
    accum_L_fn = 0.0
    opt_g.zero_grad(set_to_none=True)
    opt_d.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for b_idx, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)  # [B,3,256,256] float [-1,1]

            # ---- Generator forward + identity-embedding losses
            x_prime = G(batch)
            # Align both clean and perturbed via the same RetinaFace landmarks of the CLEAN image
            # (paper just says "FR on I_s and I_s'"; aligning to a common frame is the natural choice).
            with torch.no_grad():
                boxes, lnds = face_detection(batch, retinaface, device)
            clean_112, _ = differentiable_warping(device, batch, b_boxs=boxes, b_lnds=lnds,
                                                  affine_arrays=None, mi_transform=False,
                                                  target_shape=(112, 112))
            pert_112, _ = differentiable_warping(device, x_prime, b_boxs=boxes, b_lnds=lnds,
                                                 affine_arrays=None, mi_transform=False,
                                                 target_shape=(112, 112))
            clean_160, _ = differentiable_warping(device, batch, b_boxs=boxes, b_lnds=lnds,
                                                  affine_arrays=None, mi_transform=False,
                                                  target_shape=(160, 160))
            pert_160, _ = differentiable_warping(device, x_prime, b_boxs=boxes, b_lnds=lnds,
                                                 affine_arrays=None, mi_transform=False,
                                                 target_shape=(160, 160))

            with torch.no_grad():
                e_clean_arc = arc_embed(arc, clean_112)
                e_clean_fn = facenet_embed(facenet, clean_160)
            e_pert_arc = arc_embed(arc, pert_112)
            e_pert_fn = facenet_embed(facenet, pert_160)

            L_arc = identity_cos_loss(e_clean_arc, e_pert_arc)
            L_fn = identity_cos_loss(e_clean_fn, e_pert_fn)

            # Accumulate micro-batch losses; DLW is updated once per effective batch.
            accum_L_arc += L_arc.detach().item()
            accum_L_fn += L_fn.detach().item()
            is_accum_end = ((b_idx + 1) % accum == 0)
            if is_accum_end:
                weights = dlw.step(epoch, [accum_L_arc / accum, accum_L_fn / accum])
                accum_L_arc = 0.0
                accum_L_fn = 0.0
            else:
                # Reuse last weights within an accumulation window. First-batch fallback matches
                # the post-normalisation scheme (loss.py:81 sums to c=2 ⇒ each weight ≈ 1.0).
                if not hasattr(dlw, '_last_w') or dlw._last_w is None:
                    weights = [1.0, 1.0]
                else:
                    weights = dlw._last_w
            dlw._last_w = weights
            L_id = weights[0] * L_arc + weights[1] * L_fn

            # L_MSE: F.mse_loss on [-1, 1] domain (mean of squared pixel differences).
            # The literal L2-norm form (||·||_2) tested in v9 collapsed to a cos=1.0
            # saddle. F.mse_loss is ~10× weaker in raw magnitude but has a smoother
            # gradient profile (linear in diff), avoiding the saddle.
            L_mse = F.mse_loss(x_prime, batch)
            L_lp = lpips_loss(batch, x_prime).mean()
            L_Dg = generator_gan_loss(D, x_prime)

            L_G = (LAMBDA_ID * L_id + LAMBDA_MSE * L_mse
                   + LAMBDA_LPIPS * L_lp + LAMBDA_D * L_Dg) / accum
            # Freeze D's params during L_G backward so the generator's adversarial term
            # doesn't leak "predict-fake-as-real" gradients onto D — without this, the
            # opposing L_D gradients largely cancel and D stays near random init.
            D.requires_grad_(False)
            L_G.backward()
            D.requires_grad_(True)

            # ---- Discriminator step (also accumulated)
            L_D = discriminator_loss(D, batch, x_prime) / accum
            L_D.backward()

            if is_accum_end:
                opt_g.step()
                opt_d.step()
                opt_g.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)
                effective_iter += 1

            if global_iter % 50 == 0:
                elapsed = time.time() - t0
                print(f'[ep {epoch:02d} it {b_idx:04d}/{len(loader)} eff={effective_iter}] '
                      f'L_G={L_G.item():.4f}  L_id={L_id.item():.4f} '
                      f'(arc={L_arc.item():.3f}, fn={L_fn.item():.3f}, w={weights}) '
                      f'L_mse={L_mse.item():.4f} L_lp={L_lp.item():.3f} L_D={L_D.item():.3f} '
                      f'elapsed={elapsed/60:.1f}m')
            global_iter += 1

        ck = os.path.join(args.save_dir, f'generator_epoch_{epoch:02d}.pt')
        torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'epoch': epoch}, ck)
        print(f'[nullswap] saved {ck}')
    final = os.path.join(args.save_dir, 'generator_final.pt')
    torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'epoch': args.epochs - 1}, final)
    print(f'[nullswap] saved {final}')


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../local_datasets/hgd/')
parser.add_argument('--num_train', type=int, default=28000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--lr_g', type=float, default=5e-4)
parser.add_argument('--lr_d', type=float, default=1e-4)
parser.add_argument('--save_dir', type=str, default='./ckpts')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='Gradient accumulation steps. Effective batch = batch_size * accum_steps.')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--amp', action='store_true')
args = parser.parse_args()
main(args)
