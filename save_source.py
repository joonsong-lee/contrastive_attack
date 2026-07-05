try:
    import ml_dtypes as _mld
    for _m in ('float4_e2m1fn','float6_e2m3fn','float6_e3m2fn','float8_e8m0fnu'):
        if not hasattr(_mld,_m):
            setattr(_mld,_m,_mld.float8_e4m3fn)
except Exception:
    pass

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from pair_index import build_pair_index, build_lfw_multi_pair_index
from util import save_multi


def main(args):
    if args.dataset == 'lfw':
        full_ds = load_dataset('logasja/lfw', split='train')
        labels = full_ds['label']
        # n_pairs=5 mirrors the run.py default for the LFW OOD eval.
        multi_map = build_lfw_multi_pair_index(labels, n_pairs=5)
        paired_indices = sorted(multi_map.keys())[:args.num_samples]
    else:
        pair_map = build_pair_index(args.image_list, args.identity_file)
        ds_train = load_dataset("korexyz/celeba-hq-256x256", split='train',
                                cache_dir='./local_datasets/hgd/')
        ds_val = load_dataset("korexyz/celeba-hq-256x256", split='validation',
                              cache_dir='./local_datasets/hgd/')
        full_ds = concatenate_datasets([ds_train, ds_val])
        n_total = len(full_ds)
        src_start = args.hq_offset
        paired_indices = []
        for h in range(src_start, n_total):
            if h in pair_map:
                paired_indices.append(h)
            if len(paired_indices) >= args.num_samples:
                break
    print(f"[pair_index] {len(paired_indices)} paired samples loaded (of {args.num_samples} requested), dataset={args.dataset}.")

    def transform(eg):
        c = Compose([PILToTensor()])
        if args.dataset == 'lfw':
            eg['image'] = [torch.nn.functional.interpolate(
                               c(image).float().unsqueeze(0), size=(256, 256),
                               mode='bilinear', align_corners=True).squeeze(0)
                           for image in eg['image']]
        else:
            eg['image'] = [c(image).float() for image in eg['image']]
        return eg
    full_ds = full_ds.with_transform(transform)

    src_ds = full_ds.select(paired_indices)

    def collate_fn(examples):
        return torch.stack([e['image'] for e in examples])

    src_loader = DataLoader(src_ds, collate_fn=collate_fn, batch_size=args.batch_size)
    os.makedirs(args.save_path, exist_ok=True)

    for index, batch in enumerate(tqdm(src_loader, total=len(src_loader))):
        save_multi(batch, index, batch.shape[0], args.save_path)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--save_path", type=str, default="./valid_set_paired")
parser.add_argument("--image_list", type=str, default="./meta/image_list.txt")
parser.add_argument("--identity_file", type=str, default="./meta/identity_CelebA.txt")
parser.add_argument("--hq_offset", type=int, default=28000)
parser.add_argument("--dataset", type=str, default="celebahq", choices=["celebahq", "lfw"])
args = parser.parse_args()
main(args)
