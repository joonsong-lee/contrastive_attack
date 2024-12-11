import torch
import torch.nn as nn
from torchvision import models,transforms
import sys
sys.path.append("./STD_GAN/")

from STD_GAN.global_setting import MODEL_ROOT_DIR
import os
from STD_GAN import torchlib,config
import STD_GAN.model as stdgan

class std_gan():
    def __init__(self,device):
        self.device = device
        self.cfg = config.get_config('001')
        self.whole_model = stdgan.WholeModel(self.cfg).to(device)
        ckpt = torch.load(os.path.join(os.getcwd(),'STD_GAN','checkpoints','0060000.ckpt'))
        self.whole_model.G.load_state_dict(ckpt['Model_G'], False)
        self.whole_model.D.load_state_dict(ckpt['Model_D'], True)
    def augment_full(self,batch):
        device = self.device
        x0 = (batch.detach().clone()/255)*2-1
        x0 = x0.to(device)
        whole_model = self.whole_model
        att_list = [[0],[1],[2],[3]]
        ress = []
        cfg = self.cfg
        for att in att_list:
            sample_y_fake = torch.zeros(x0.size(0), len(cfg.use_atts)).type_as(x0)
            for a in att:
                sample_y_fake[:, a] = 1
            sample_z = stdgan.generate_z(cfg, sample_y_fake)
            stdgan.mask_z(cfg, sample_y_fake, sample_z)
            res = whole_model(sample_y_fake, sample_z, x0, None, mask='test')
            ress.append( res)
        ress = torch.cat(ress)
        ress = ress.div_(2).add_(0.5).mul_(255)
        ress = ress.clamp(0,255)
        return ress
