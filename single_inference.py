import sys
sys.path.append('./AdaFace/')
sys.path.append('./arcface_torch')
#from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.transforms import Compose,PILToTensor
from tqdm import tqdm
import numpy as np
import cv2 as cv
import argparse
from arcface_torch.backbones.iresnet import iresnet50
from attack import *
import os
from util import torch_to_cv,save_multi
from facenet_pytorch import InceptionResnetV1
import AdaFace.net as adanet
from collections import namedtuple

def load_single_img(lpath):
    i_t = cv.imread(lpath)
    i_t = cv.cvtColor(i_t,cv.COLOR_BGR2RGB)
    i_t = np.transpose(i_t,(2,0,1))
    i_t = torch.from_numpy(i_t).float()
    return i_t.unsqueeze(0)
def save_single(img,pertur,savepath):
    res = (img+pertur).clamp(0,255)
    res = res.squeeze(0).detach().cpu().numpy()
    res = np.transpose(res,(1,2,0))
    res = np.uint8(res)
    res = cv.cvtColor(res,cv.COLOR_RGB2BGR)
    cv.imwrite(savepath,res)
def main(args):
    batch_size = args.batch_size
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
    nets = nn.ModuleDict({"arc":resnet2,"arc2":resnet,"ada":ada,"facenet":facenet})
    os.makedirs(args.save_path,exist_ok=True)
    img = load_single_img(args.ipath)
    img = torchvision.transforms.functional.resize(img,(256,256))
    if args.victim is not None:
        victim = load_single_img(args.victim)
        victim = torchvision.transforms.functional.resize(victim,(256,256))
        if(args.attack_method == "rec_ensemble"):
            attack = direct_attack(device,nets,lpips_weight=args.lpips_weight)
            batch = torch.cat((img,victim),dim=0)
            res = attack.attack(batch,2,iter=args.iter,eps=args.eps,lr=args.lr,img_size=args.img_size)
            pertur = res[0].clone()
        elif(args.attack_method =="cont"):
            attack = contrastive_opposite(device,nets,lpips_weight=args.lpips_weight)
            pertur = attack.attack(img,victim,1,iter=args.iter,eps=args.eps,lr=args.lr,img_size=args.img_size)
    else:
        attack = contrastive_opposite(device,nets,lpips_weight=args.lpips_weight)
        pertur = attack.attack(img,img,1,iter=args.iter,eps=args.eps,lr=args.lr,img_size=args.img_size)
    
    save_single(img.to(device),pertur.to(device),os.path.join(args.save_path,args.file_name))



parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")          
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--iter", type=int, default=300)  
parser.add_argument("--eps", type=int, default=16)  
parser.add_argument("--lr", type=float, default=0.001)  
parser.add_argument("--img_size", type=int, default=256)  
parser.add_argument("--attack_method", type=str, default="cont_opposite")  
parser.add_argument("--save_path", type=str, default="/data/ljsong7/gradpj/debug")
parser.add_argument("--file_name", type=str, default="asset.jpg")
parser.add_argument("--ipath", type=str, default="/data/ljsong7/gradpj/musk.jpg") 
parser.add_argument("--victim", type=str, default=None,
                    help="For --attack_method=cont: a second photo of the SAME identity as --ipath (used as the negative-pair reference). For rec_ensemble: the victim identity image.")
parser.add_argument("--lpips_weight", type=float, default=0.1)
args = parser.parse_args()
main(args)
