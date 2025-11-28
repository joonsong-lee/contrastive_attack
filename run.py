import sys
sys.path.append('./AdaFace/')
import AdaFace.net as adanet
sys.path.remove('./AdaFace/')
sys.path.append('./arcface_torch')
from arcface_torch.backbones.iresnet import iresnet50
sys.path.remove('./arcface_torch')
sys.path.append('./RetinaFace_Pytorch')
from RetinaFace_Pytorch import torchvision_model
sys.path.remove('./RetinaFace_Pytorch')
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

from collections import namedtuple

def main(args):
    
    ds = load_dataset("korexyz/celeba-hq-256x256",split='validation[:1000]',cache_dir='./local_datasets/hgd/')
    def collate_fn(examples):
        images = []
        for example in examples:
            images.append((example['image']))
            
        pixel_values = torch.stack(images)
        return pixel_values

    
    def transform(eg):
        c = Compose([PILToTensor(),])
        eg['image']= [c(image).float() for image in eg['image']]
        return eg
    ds= ds.with_transform(transform)
    batch_size = args.batch_size
    dataloader = DataLoader(ds, collate_fn=collate_fn, batch_size=batch_size)
    device = torch.device(args.device)
    resnet = iresnet50().to(device)
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    retinaface = torchvision_model.create_retinaface(return_layers)
    retina_dict = retinaface.state_dict()
    pre_state_dict = torch.load(os.path.join('./RetinaFace_Pytorch/model.pt'))
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    retinaface.load_state_dict(pretrained_dict)
    retinaface = retinaface.to(device)
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
    attack = contrastive_opposite(retinaface,device,nets)
    os.makedirs(args.save_path,exist_ok=True)
    
    for index,batch in enumerate(tqdm(dataloader)):
        pertur = attack.attack(batch,batch.shape[0],iter=args.iter,eps=args.eps,lr=args.lr,img_size=args.img_size)
        save_multi(batch.to(device)+pertur.to(device),index,batch.shape[0],args.save_path)



parser = argparse.ArgumentParser()     
parser.add_argument("--device", type=str, default="cuda:0")          
parser.add_argument("--batch_size", type=int, default=16)           
parser.add_argument("--iter", type=int, default=300)  
parser.add_argument("--eps", type=int, default=8)  
parser.add_argument("--lr", type=float, default=0.001)  
parser.add_argument("--img_size", type=int, default=256)  
parser.add_argument("--attack_method", type=str, default="cont_opposite")  
parser.add_argument("--save_path", type=str, default="./out")  
args = parser.parse_args()
main(args)
