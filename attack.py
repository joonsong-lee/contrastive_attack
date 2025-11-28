from util import differentiable_warping

import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from RetinaFace_Pytorch import eval_widerface
from augmen import std_gan
import gc



def face_detection(batch_o,retinaface,device):
    boxes,scores,landmarks = [],[],[]
    batch = (batch_o*0.5+0.5)*255.
    img_h, img_w = batch.shape[2], batch.shape[3]
    with torch.no_grad():
        picked_boxes, picked_landmarks, picked_scores = eval_widerface.get_detections(batch, retinaface, score_threshold=0.5, iou_threshold=0.3)
        for b,s,l in zip(picked_boxes,picked_scores,picked_landmarks):
            if(s==None):
                print('nodetect_retina')
                box = torch.tensor([0,0,img_w, img_h],dtype=torch.float32).to(device)
                landmark = torch.tensor([
                    [img_w * 0.3, img_h * 0.3], # Left Eye
                    [img_w * 0.7, img_h * 0.3], # Right Eye
                    [img_w * 0.5, img_h * 0.5], # Nose
                    [img_w * 0.3, img_h * 0.7], # Left Mouth
                    [img_w * 0.7, img_h * 0.7]  # Right Mouth
                ], dtype=torch.float32).to(device)
                boxes.append(box)
                landmarks.append(landmark)
            else :
                ss = torch.squeeze(s,dim=0)
                bb = torch.squeeze(b,dim=0)
                if(ss.shape[0] >1) :
                    boxes.append(bb[torch.argmax(ss)].squeeze(0))
                    landmarks.append(l[torch.argmax(ss)].squeeze(0).view(5,2))

                else :
                    boxes.append(bb.squeeze(0))
                    landmarks.append(l.squeeze(0).view(5,2))
    boxes = torch.stack(boxes,dim=0)
    landmarks = torch.stack(landmarks,dim=0)
    
    return boxes,landmarks


class contrastive_opposite(nn.Module):
    def __init__(self,retina,device,nets):
        super(contrastive_opposite,self).__init__()
        self.mse_loss = nn.MSELoss()
        self.csloss = nn.CosineEmbeddingLoss()
        self.retinaface = retina
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.nets = nets.to(device)
        self.sg = std_gan(device)
        self.ce = nn.CrossEntropyLoss().to(self.device)
        self.kernel = self.gkern(klen=15, nsig=3)

    def gkern(self,klen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, klen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = np.stack([kernel, kernel, kernel])
        kernel= np.expand_dims(kernel,1)
        kernel = torch.FloatTensor(kernel).to(self.device)
        return kernel

    def rgb2bgr(self,imgs):
        bgrims = torch.flip(imgs,[1])
        return bgrims

    def cont_loss(self,feat,target):
        batch_size = feat.shape[0]
        s = torch.norm(feat,dim=1).mean() 
        target_size = target.shape[0]//feat.shape[0]
        labels = torch.cat([torch.arange(batch_size) for i in range(target_size)], dim=0).to(self.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels[:batch_size]
        sim = F.linear(feat,target)
        logit = sim[labels.bool()].view(labels.shape[0],-1).to(self.device)
        logit = logit * s
        labels = torch.zeros(logit.shape[0],dtype = torch.long).to(self.device)
        loss = self.ce(logit,labels)
        return loss

    def flow_cont(self,b_t,net,ort,ori_feat,aug_feat,batch_size,is_ada):
        if(is_ada):
            ada_bt = self.rgb2bgr(b_t)
            feat,_ = net(ada_bt)
        else:
            feat = net(b_t)
        target = torch.cat((ort,ori_feat,aug_feat),dim=0)
        feat = torch.cat([feat[:batch_size],feat[batch_size:]],dim=1)
        feat = F.normalize(feat)
        loss = self.cont_loss(feat,target)
        return loss

    def ready_made_attack(self,batch):
        with torch.no_grad():
            augmented = self.sg.augment_full(batch).to(self.device)
            b_boxs,b_lnds = face_detection(batch,self.retinaface,self.device)
            aug_box,aug_lnds = face_detection(augmented,self.retinaface,self.device)
        blob,affines = differentiable_warping(self.device,batch, b_boxs,b_lnds=b_lnds,angle=None,affine_arrays=None,mi_transform=False)
        blob=torch.cat([blob,blob.fliplr()],dim=0)
        blob.div_(255).sub_(0.5).div_(0.5)
        feat_list,ort_list = [],[]
        for key in self.nets.keys():
            if(key=='ada'):
                ada_blob = self.rgb2bgr(blob)
                feat,_ = self.nets[key](ada_blob)
            else: feat = self.nets[key](blob)
            feat = torch.cat([feat[:batch.shape[0]],feat[batch.shape[0]:]],dim=1)
            feat_list.append(feat.detach().clone())
        for feat in feat_list:
            ort = -1*feat      # +(torch.randn(feat.shape).to(device)/feat.sum())/feat.shape[0]
            ort = F.normalize(ort)
            feat = F.normalize(feat)
            ort_list.append(ort.detach().clone())
        #del blob,ada_blob,feat,ort
        aug_blob,_ = differentiable_warping(self.device,augmented, aug_box,b_lnds=aug_lnds,angle=None,affine_arrays=None,mi_transform=False)
        aug_blob=torch.cat([aug_blob,aug_blob.fliplr()],dim=0)
        aug_blob = aug_blob.div_(255).sub_(0.5).div_(0.5)
        aug_feats = []
        for key in self.nets.keys():
            if(key=='ada'):
                ada_blob = self.rgb2bgr(aug_blob)
                feat,_ = self.nets[key](ada_blob)
            else: feat = self.nets[key](aug_blob)
            feat = torch.cat([feat[:batch.shape[0]*4],feat[batch.shape[0]*4:]],dim=1)
            feat = F.normalize(feat)
            aug_feats.append(feat.detach().clone())

        return ort_list,feat_list,aug_feats,affines


    def attack(self,batch_o,batch_size=32,img_size=256,iter=500,lr=0.001,eps=16):
        new_shape = int(112)
        batch = ((batch_o/255. - 0.5) /0.5).to(self.device)
        ort_list,feat_list,aug_feats,affines = self.ready_made_attack(batch)
        g = torch.zeros_like(batch, requires_grad=True).to(self.device)
        delta = torch.zeros_like(batch,requires_grad=True).to(self.device)
        torch.cuda.empty_cache()
        for k in range (iter) :
            idx = 0
            b_t = batch + delta
            b_t, _ = differentiable_warping(self.device,b_t, b_boxs=None,b_lnds=None,angle=None,affine_arrays=affines,mi_transform=True)
            b_t = torch.cat([b_t,b_t.fliplr()],dim=0)
            loss = torch.zeros(len(self.nets))
            for key in self.nets.keys():
                if(key =='ada'):
                    loss[idx]=self.flow_cont(b_t,self.nets[key],ort_list[idx],feat_list[idx],aug_feats[idx],batch_size,True)
                    idx+=1
                else:
                    loss[idx]=self.flow_cont(b_t,self.nets[key],ort_list[idx],feat_list[idx],aug_feats[idx],batch_size,False)
                    idx+=1
            constraint = self.mse_loss(batch,batch+delta)
            out = loss.mean() + 0.1*constraint
            #print(out)
            out.backward()
            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c,self.kernel,padding=15//2,groups=3)
            norm_grad = torch.norm(grad_c,p=1,dim=(1,2,3),keepdim=True)
            g = g + grad_c/norm_grad
            delta.grad.zero_()
            delta.data = delta.data - lr *torch.sign(g)
            delta.data = delta.data.clamp(-eps/127 ,eps/127)
            delta.data = ((batch + delta.data)).clamp(-1,1) - batch
            delta.data = delta.data
        return delta.data * 127.5

