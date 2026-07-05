try:
    import ml_dtypes as _mld
    for _m in ('float4_e2m1fn','float6_e2m3fn','float6_e3m2fn','float8_e8m0fnu'):
        if not hasattr(_mld,_m):
            setattr(_mld,_m,_mld.float8_e4m3fn)
except Exception:
    pass
import sys
sys.path.append('./AdaFace/')
import AdaFace.net as adanet
sys.path.remove('./AdaFace/')
sys.path.append('./arcface_torch')
from arcface_torch.backbones.iresnet import iresnet50
sys.path.remove('./arcface_torch')
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.transforms import Compose,PILToTensor
from tqdm import tqdm
import numpy as np
import cv2 as cv
import argparse

from attack import *
import os
from util import torch_to_cv,save_multi
from facenet_pytorch import InceptionResnetV1
from pair_index import build_pair_index, build_multi_pair_index, build_lfw_multi_pair_index

from collections import namedtuple

def main(args):
    n_pairs = args.n_pairs
    if args.dataset == 'lfw':
        full_ds = load_dataset('logasja/lfw', split='train')
        labels = full_ds['label']
        # LFW uses HQ-style label index pairing; n_pairs always uses the multi-pair builder.
        multi_map = build_lfw_multi_pair_index(labels, n_pairs=max(n_pairs, 1))
        src_keys = sorted(multi_map.keys())
        paired_indices = src_keys[:args.num_samples]
        partner_indices_list = [[multi_map[h][pi] for h in paired_indices] for pi in range(n_pairs)]
    else:
        if n_pairs > 1:
            multi_map = build_multi_pair_index(args.image_list, args.identity_file, n_pairs=n_pairs)
        else:
            single_map = build_pair_index(args.image_list, args.identity_file)
        from datasets import concatenate_datasets
        ds_train = load_dataset("korexyz/celeba-hq-256x256",split='train',cache_dir='./local_datasets/hgd/')
        ds_val = load_dataset("korexyz/celeba-hq-256x256",split='validation',cache_dir='./local_datasets/hgd/')
        full_ds = concatenate_datasets([ds_train, ds_val])
        n_total = len(full_ds)
        src_start = args.hq_offset
        paired_indices = []
        partner_indices_list = [[] for _ in range(n_pairs)]
        for h in range(src_start, n_total):
            if n_pairs > 1:
                if h in multi_map:
                    paired_indices.append(h)
                    for pi in range(n_pairs):
                        partner_indices_list[pi].append(multi_map[h][pi])
            else:
                if h in single_map:
                    paired_indices.append(h)
                    partner_indices_list[0].append(single_map[h])
            if len(paired_indices) >= args.num_samples:
                break
    print(f"[pair_index] {len(paired_indices)} paired samples loaded (of {args.num_samples} requested), n_pairs={n_pairs}, dataset={args.dataset}.")

    def transform(eg):
        c = Compose([PILToTensor(),])
        # LFW is 250x250; resize to 256x256 to match the rest of the pipeline.
        if args.dataset == 'lfw':
            eg['image']= [torch.nn.functional.interpolate(
                              c(image).float().unsqueeze(0), size=(256, 256),
                              mode='bilinear', align_corners=True).squeeze(0)
                          for image in eg['image']]
        else:
            eg['image']= [c(image).float() for image in eg['image']]
        return eg
    full_ds = full_ds.with_transform(transform)

    src_ds = full_ds.select(paired_indices)
    pair_datasets = [full_ds.select(pidx) for pidx in partner_indices_list]

    def collate_fn(examples):
        return torch.stack([e['image'] for e in examples])

    batch_size = args.batch_size
    src_loader = DataLoader(src_ds, collate_fn=collate_fn, batch_size=batch_size)
    pair_loaders = [DataLoader(pds, collate_fn=collate_fn, batch_size=batch_size) for pds in pair_datasets]
    device = torch.device(args.device)
    resnet = iresnet50().to(device)
    resnet.load_state_dict(torch.load('./arcface_torch/backbone.pth'))
    resnet.eval()
    ada = adanet.build_model('ir_50')
    statedict = torch.load('./AdaFace/adaface_ir50_webface4m.ckpt')['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    ada.load_state_dict(model_statedict)
    ada = ada.eval().to(device)
    resnet2 = iresnet50().to(device)
    resnet2.load_state_dict(torch.load('./arcface_torch/gli_backbone.pth'))
    resnet2.eval()

    facenet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    nets = nn.ModuleDict({"arc":resnet2,"arc2":resnet,"ada":ada})#,"facenet":facenet})

    retinaface = None
    if args.alignment == 'retina5p':
        sys.path.append('./RetinaFace_Pytorch')
        import torchvision_model as _rf_tvm
        return_layers = {'layer2':1,'layer3':2,'layer4':3}
        retinaface = _rf_tvm.create_retinaface(return_layers)
        retina_dict = retinaface.state_dict()
        pre_state_dict = torch.load('./RetinaFace_Pytorch/model.pt', map_location='cpu')
        pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
        retinaface.load_state_dict(pretrained_dict)
        retinaface = retinaface.to(device).eval()
        for p in retinaface.parameters():
            p.requires_grad_(False)
        print(f'[run] RetinaFace loaded for alignment=retina5p')

    attack = contrastive_opposite(device,nets,lpips_weight=args.lpips_weight,retinaface=retinaface,loss_type=args.loss_type,grad_smooth=args.grad_smooth,regularizer=args.regularizer,reg_weight=args.reg_weight)
    #attack = direct_attack(device,nets,lpips_weight=args.lpips_weight)
    os.makedirs(args.save_path,exist_ok=True)

    use_alignment = (args.alignment == 'retina5p')
    all_loaders = [src_loader] + pair_loaders
    for index, batches in enumerate(tqdm(zip(*all_loaders), total=len(src_loader))):
        batch = batches[0]
        pair_batches = list(batches[1:]) if n_pairs > 1 else batches[1]
        pertur = attack.attack(batch, pair_batches, batch.shape[0], iter=args.iter, eps=args.eps, lr=args.lr, img_size=args.img_size, use_alignment=use_alignment)
        save_multi(batch.to(device)+pertur.to(device), index, batch.shape[0], args.save_path)



parser = argparse.ArgumentParser()     
parser.add_argument("--device", type=str, default="cuda:0")          
parser.add_argument("--batch_size", type=int, default=16)           
parser.add_argument("--iter", type=int, default=300)  
parser.add_argument("--eps", type=int, default=8)  
parser.add_argument("--lr", type=float, default=0.001)  
parser.add_argument("--img_size", type=int, default=256)  
parser.add_argument("--attack_method", type=str, default="cont_opposite")
parser.add_argument("--save_path", type=str, default="./out")
parser.add_argument("--lpips_weight", type=float, default=0.1)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--image_list", type=str, default="./contrastive_attack/meta/image_list.txt")
parser.add_argument("--identity_file", type=str, default="./contrastive_attack/meta/identity_CelebA.txt")
parser.add_argument("--hq_offset", type=int, default=28000,
                    help="Offset mapping HF 'validation' split local index -> global CelebA-HQ index. Verify against your dataset split.")
parser.add_argument("--alignment", type=str, default="vanilla", choices=["vanilla","retina5p"],
                    help="vanilla: F.interpolate(256->112) as before. retina5p: RetinaFace-Pytorch 5-point differentiable similarity warp to a canonical 112x112 crop.")
parser.add_argument("--loss_type", type=str, default="ce", choices=["ce","supcon"],
                    help="ce: cross-entropy with single positive (ort). supcon: SupCon with multi-positive masked log prob.")
parser.add_argument("--n_pairs", type=int, default=1,
                    help="Number of same-identity pair images per source. >1 uses build_multi_pair_index.")
parser.add_argument("--grad_smooth", type=str, default="none", choices=["none","ti"],
                    help="none: no gradient smoothing. ti: Gaussian-kernel smoothing before sign step (TI-FGSM).")
parser.add_argument("--regularizer", type=str, default="lpips", choices=["lpips","wavelet","ssim","none"],
                    help="Imperceptibility regularizer: lpips | wavelet (DWT low-freq penalty) | ssim (1-SSIM) | none.")
parser.add_argument("--reg_weight", type=float, default=None,
                    help="Regularizer weight. If omitted, falls back to --lpips_weight.")
parser.add_argument("--dataset", type=str, default="celebahq", choices=["celebahq", "lfw"],
                    help="Source dataset: celebahq (default, uses meta/image_list.txt) or lfw (logasja/lfw via HF).")
args = parser.parse_args()
main(args)
