import torch
from util import get_rot_angle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from RetinaFace_Pytorch import eval_widerface
from augmen import std_gan
import gc



def face_detection(batch,retinaface,device):
    boxes,scores,landmarks = [],[],[]
    with torch.no_grad():
        picked_boxes, picked_landmarks, picked_scores = eval_widerface.get_detections(batch, retinaface, score_threshold=0.5, iou_threshold=0.3)
        for b,s,l in zip(picked_boxes,picked_scores,picked_landmarks):
            if(s==None):
                print('nodetect_retina')
                box = torch.tensor([0,0,250,250]).to(device)
                landmark = torch.randint(0,250,(5,2)).float().to(device)
                landmark[0][0],landmark[0][1],landmark[1][0],landmark[1][1] = 10,10,10,50
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



class Base():
    def __init__(self,retina,device):
        self.mse_loss = nn.MSELoss()
        self.csloss = nn.CosineEmbeddingLoss()
        self.retinaface = retina
        self.device = device
        self.ce = nn.CrossEntropyLoss()

    def add_margin(self,logit,t_idx,margin=0.5,s=64.0):
        idx = torch.arange(logit.shape[0]).view(-1)
        with torch.no_grad():
            t_logit = logit[idx,t_idx].clone()
            t_logit.arccos_()
            t_logit = t_logit+margin
            t_logit.cos_()
            logit[idx,t_idx] = t_logit
        logit = logit * s
        return logit
    
    def margin_loss (self,feat,ort,batch_size,ll_idx):
        norm_feat= F.normalize(feat)
        logit = F.linear(norm_feat,ort)
        t_idx = torch.arange(logit.shape[0]).view(-1).to(self.device)
        logit = self.add_margin(logit,t_idx)
        t_logit = torch.gather(logit,1,t_idx.unsqueeze(1))
        assert (torch.isnan(t_logit)[0]==False),"There is nan value in loigt"
        loss1 = (-1*t_logit).mean()
        loss3 = ((torch.var(norm_feat,dim=1) - torch.var(ort,dim=1))**2).mean()
        loss4 = ((torch.mean(norm_feat,dim=1)-torch.mean(ort,dim=1))**2).mean()
        out = loss1+ 0.5*loss3+loss4
        return out



    
    def make_blob (self,batch, b_boxs,b_lnds=None,angle=np.array([999])):
        blob1 = torch.zeros((batch.shape[0],3,112,112)).to(self.device)
        blob2 = torch.zeros((batch.shape[0],3,112,112)).to(self.device)
        angles = np.zeros(batch.shape[0])
        for b in range(batch.shape[0]): 
            if(angle[0] == 999) : 
                angle_b = get_rot_angle(b_lnds[b][0],b_lnds[b][1])
                angles[b] = angle_b
            else: 
                angle_b = angle[b]
            c_img = batch[b][:,int(b_boxs[b][1]):int(b_boxs[b][3]),int(b_boxs[b][0]):int(b_boxs[b][2])]
            blob = torchvision.transforms.functional.rotate(c_img,angle=angle_b)
            blob = torchvision.transforms.functional.resize(blob,(112,112))
            blob1[b] = blob.clone()
            blob2[b] = blob.clone()
        blob2 = blob2.fliplr()
        return torch.cat((blob1,blob2),dim=0),angles
    
    def find_ll(self,feat):
        ll_idx = np.zeros(feat.shape[0])
        for i in range(feat.shape[0]):
            min_sim = 1
            min_loc = 0
            for j in range(feat.shape[0]):
                if(i==j) : continue
                elif(F.cosine_similarity(feat[i],feat[j],dim=0)<min_sim):
                    min_sim =F.cosine_similarity(feat[i],feat[j],dim=0)
                    min_loc = j
            ll_idx[i]=min_loc
        return ll_idx.astype(int)
    
    def permute_with_idx(self,feat,ll_idx):
        permuted = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            permuted[i] = feat[ll_idx[i]].detach().clone()
        return permuted
    
    def permute_victim(self,feat,ll_idx):
        permuted = self.permute_with_idx(feat,ll_idx)
        return permuted
    


    def rgb2bgr(self,imgs):
        bgrims = torch.flip(imgs,[1])
        return bgrims
    def flow(self,b_t,net,ort,batch_size,is_ada,prt):
        if(is_ada):
            ada_bt = self.rgb2bgr(b_t)
            feat,_ = net(ada_bt)
        else:
            feat = net(b_t)
        feat = torch.cat([feat[:batch_size*2],feat[batch_size*2:]],dim=1)
        loss = self.cal_loss(feat,ort,batch_size)
        if(prt%20==0):
            print(torch.norm(feat.grad))

        return loss
    def flow_margin(self,b_t,net,ort,batch_size,is_ada,ll_idx):
        if(is_ada):
            ada_bt = self.rgb2bgr(b_t)
            feat,_ = net(ada_bt)
        else:
            feat = net(b_t)
        feat = torch.cat([feat[:batch_size*2],feat[batch_size*2:]],dim=1)
        loss = self.margin_loss(feat[:batch_size],ort,batch_size,ll_idx) + self.margin_loss(feat[batch_size:],ort,batch_size,ll_idx)
        loss2 = F.cosine_similarity(feat[:batch_size],feat[batch_size:],dim=1).mean()
        return loss/2 - loss2*0.001

    

    def margin_single_attack(self,net,batch,batch_size,ort,angle,b_boxs,ll_idx,iter=250,is_bgr=False,img_size=256,lr=0.001,eps=8):
        device = self.device
        new_shape = int(112)
        ran_crop = torchvision.transforms.RandomResizedCrop((img_size,img_size),scale=(0.85,0.95))
        delta = torch.zeros_like(batch, requires_grad=True).to(device)
        g = torch.zeros_like(batch, requires_grad=True).to(device)
        retinaface = self.retinaface.to(device)
        for k in range (iter) :
            b_crop = ran_crop(batch)
            b_crop_mul = b_crop.detach().clone()
            c_boxs,_ = face_detection(b_crop_mul.mul_(0.5).add_(0.5).mul_(255),retinaface,device)
            b_t = torch.cat([batch + delta,b_crop + delta],dim=0)
            box_a = torch.cat((b_boxs,c_boxs),dim=0)
            b_t,_ = self.make_blob(b_t,box_a,angle=np.concatenate((angle,angle),axis=0))
            p = torch.randint(low = int(new_shape * 0.8), high = new_shape,size = (1,))[0]
            b_t = torchvision.transforms.functional.resize(b_t,(p,p))
            rest = new_shape-p
            top = np.random.randint(low= 0, high=rest)
            left = np.random.randint(low= 0, high=rest)
            b_t= F.pad(b_t,(left,rest-left,top,rest-top),mode='constant',value=0)
            if(is_bgr):
                bgr_bt = self.rgb2bgr(b_t)
                feat,_ = net(bgr_bt)
                
            else:feat = net(b_t)
            feat = torch.cat([feat[:batch_size*2],feat[batch_size*2:]],dim=1)
            loss = self.margin_loss(feat[:batch_size],ort,batch_size,ll_idx) + self.margin_loss(feat[batch_size:],ort,batch_size,ll_idx)
            loss2 = F.cosine_similarity(feat[:batch_size],feat[batch_size:],dim=1).mean()
            constraint = self.mse_loss(batch,batch+delta)
            out = loss/2-0.001*loss2+ 0.1*constraint
            print(out)
            out.backward()
            grad_c = delta.grad.clone()
            norm_grad = torch.norm(grad_c,p=1)
            g = g + grad_c/norm_grad
            delta.grad.zero_()
            delta.data = delta.data - lr *torch.sign(g)
            delta.data = delta.data.clamp(-eps/127 ,eps/127)
            delta.data = ((batch + delta.data)).clamp(-1,1) - batch
            delta.data = delta.data
        return delta.data



class contrastive_attack(Base):
    def __init__(self,retina,device,nets):
        super().__init__(retina,device)
        self.nets = nets
        self.sg = std_gan(device)
        self.ce = nn.CrossEntropyLoss().to(self.device)

    def cont_loss(self,feat,target):
        batch_size = feat.shape[0]
        s=64
        target_size = target.shape[0]//feat.shape[0]
        labels = torch.cat([torch.arange(batch_size) for i in range(target_size)], dim=0).to(self.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels[:batch_size]
        sim = F.linear(feat,target)
        t_idx = torch.arange(sim.shape[0])
        sim = self.add_margin(sim,t_idx,0.5,1)
        logit = sim[labels.bool()].view(labels.shape[0],-1).to(self.device)
        logit = logit * s
        labels = torch.zeros(logit.shape[0],dtype = torch.long).to(self.device)
        loss = self.ce(logit,labels)
        return loss

    def flow_cont(self,b_t,net,ort,ori_feat,aug_feat,batch_size,is_ada,ll_idx):
        if(is_ada):
            ada_bt = self.rgb2bgr(b_t)
            feat,_ = net(ada_bt)
        else:
            feat = net(b_t)
        target = torch.cat((ort,ori_feat,aug_feat),dim=0)
        feat = torch.cat([feat[:batch_size*2],feat[batch_size*2:]],dim=1)
        feat = F.normalize(feat)
        loss = (self.cont_loss(feat[:batch_size],target)+self.cont_loss(feat[batch_size:],target))/2
        loss2 = F.cosine_similarity(feat[:batch_size],feat[batch_size:],dim=1).mean()
        loss3 = ((torch.var(feat[:batch_size],dim=1) - torch.var(ort,dim=1))**2).mean()
        loss4 = ((torch.mean(feat[:batch_size],dim=1)-torch.mean(ort,dim=1))**2).mean()

        return loss - loss2*0.001 + loss3 + loss4

    def single_cont(self,net,batch,batch_size,ort,ori_feat,aug_feat,angle,b_boxs,ll_idx,iter=250,is_bgr=False,img_size=256,lr=0.001,eps=8):
        device = self.device
        new_shape = int(112)
        ran_crop = torchvision.transforms.RandomResizedCrop((img_size,img_size),scale=(0.85,0.95))
        delta = torch.zeros_like(batch, requires_grad=True).to(device)
        g = torch.zeros_like(batch, requires_grad=True).to(device)
        retinaface = self.retinaface.to(device)
        for k in range (iter) :
            b_crop = ran_crop(batch)
            b_crop_mul = b_crop.detach().clone()
            c_boxs,_ = face_detection(b_crop_mul.mul_(0.5).add_(0.5).mul_(255),retinaface,device)
            b_t = torch.cat([batch + delta,b_crop + delta],dim=0)
            box_a = torch.cat((b_boxs,c_boxs),dim=0)
            b_t,_ = self.make_blob(b_t,box_a,angle=np.concatenate((angle,angle),axis=0))
            p = torch.randint(low = int(new_shape * 0.8), high = new_shape,size = (1,))[0]
            b_t = torchvision.transforms.functional.resize(b_t,(p,p))
            rest = new_shape-p
            top = np.random.randint(low= 0, high=rest)
            left = np.random.randint(low= 0, high=rest)
            b_t= F.pad(b_t,(left,rest-left,top,rest-top),mode='constant',value=0)
            if(is_bgr):
                loss = self.flow_cont(b_t,net,ort,ori_feat,aug_feat,batch_size,True,ll_idx)   
            else:loss = self.flow_cont(b_t,net,ort,ori_feat,aug_feat,batch_size,False,ll_idx)
            constraint = self.mse_loss(batch,batch+delta)
            out = loss+ 0.2*constraint
            print(out)
            out.backward()
            grad_c = delta.grad.clone()
            norm_grad = torch.norm(grad_c,p=1)
            g = g + grad_c/norm_grad
            delta.grad.zero_()
            delta.data = delta.data - lr *torch.sign(g)
            delta.data = delta.data.clamp(-eps/127 ,eps/127)
            delta.data = ((batch + delta.data)).clamp(-1,1) - batch
            delta.data = delta.data
        return delta.data

    def attack(self,batch,batch_size=32,img_size=256,iter=500,lr=0.001,eps=16):
        device = self.device
        retinaface = self.retinaface.to(device)
        sg = self.sg
        batch = batch.to(device)
        new_shape = int(112)
        nets = self.nets.to(device)
        ran_crop = torchvision.transforms.RandomResizedCrop((img_size,img_size),scale=(0.8,0.9))
        g = torch.zeros_like(batch, requires_grad=True).to(device)
        with torch.no_grad():
            b_boxs,b_lnds = face_detection(batch,retinaface,device)
            augmented = sg.augment_full(batch).to(device)
        blob,angle = self.make_blob(batch,b_boxs,b_lnds)
        blob = blob.div_(255).sub_(0.5).div_(0.5)
        feat_list,ort_list = [],[]
        for key in nets.keys():
            if(key=='ada'):
                ada_blob = self.rgb2bgr(blob)
                feat,_ = nets[key](ada_blob)
            else: feat = nets[key](blob)
            feat = torch.cat([feat[:batch_size],feat[batch_size:]],dim=1)
            feat_list.append(feat.detach().clone())
        ll_idx = self.find_ll(feat_list[0])
        for feat in feat_list:
            ort = self.permute_victim(feat,ll_idx)
            ort = F.normalize(ort)
            feat = F.normalize(feat)
            ort_list.append(ort.detach().clone())
        del blob,ada_blob,feat,ort
        aug_blob,_ = self.make_blob(augmented,torch.cat((b_boxs,b_boxs,b_boxs,b_boxs),dim=0),angle=np.concatenate((angle,angle,angle,angle),axis=0))
        aug_blob = aug_blob.div_(255).sub_(0.5).div_(0.5)
        aug_feats = []
        for key in nets.keys():
            if(key=='ada'):
                ada_blob = self.rgb2bgr(aug_blob)
                feat,_ = nets[key](ada_blob)
            else: feat = nets[key](aug_blob)
            feat = torch.cat([feat[:batch_size*4],feat[batch_size*4:]],dim=1)
            feat = F.normalize(feat)
            aug_feats.append(feat.detach().clone())
        del aug_blob,ada_blob,feat,augmented
        batch = batch.div_(255).sub_(0.5).div_(0.5)
        tdelta = torch.zeros_like(batch).to(device)
        idx = 0
        for key in nets.keys():
            if(key=='ada'):
                tdelta=self.single_cont(nets[key],batch+tdelta,batch_size,ort_list[idx],feat_list[idx],aug_feats[idx],angle,b_boxs,ll_idx,iter//2,True,eps=eps/2)
                idx+=1
            else:
                tdelta=self.single_cont(nets[key],batch+tdelta,batch_size,ort_list[idx],feat_list[idx],aug_feats[idx],angle,b_boxs,ll_idx,iter//2,eps=eps/2)
                idx+=1
        delta = torch.zeros_like(batch).to(device)+tdelta.detach().clone()
        torch.cuda.empty_cache()
        delta.requires_grad_()
        for k in range (iter) :
            b_crop = ran_crop(batch)
            b_crop_mul = b_crop.detach().clone()
            c_boxs,_ = face_detection(b_crop_mul.mul_(0.5).add_(0.5).mul_(255),retinaface,device)
            b_t = torch.cat([batch + delta,b_crop + delta],dim=0)
            b_t,_ = self.make_blob(b_t,torch.cat((b_boxs,c_boxs),dim=0),angle=np.concatenate((angle,angle),axis=0))
            p = torch.randint(low = int(new_shape * 0.8), high = new_shape,size = (1,))[0]
            b_t = torchvision.transforms.functional.resize(b_t,(p,p))
            rest = new_shape-p
            top = np.random.randint(low= 0, high=rest)
            left = np.random.randint(low= 0, high=rest)
            b_t= F.pad(b_t,(left,rest-left,top,rest-top),mode='constant',value=0)
            loss = 0
            idx = 0
            loss = torch.zeros(len(nets))
            for key in nets.keys():
                if(key =='ada'):
                    loss[idx]=self.flow_cont(b_t,nets[key],ort_list[idx],feat_list[idx],aug_feats[idx],batch_size,True,ll_idx)
                    idx+=1
                else:
                    loss[idx]=self.flow_cont(b_t,nets[key],ort_list[idx],feat_list[idx],aug_feats[idx],batch_size,False,ll_idx)
                    idx+=1
            constraint = self.mse_loss(batch,batch+delta)
            out = loss.mean() + 0.1*constraint
            print(out)
            out.backward()
            grad_c = delta.grad.clone()
            norm_grad = torch.norm(grad_c,p=1)
            g = g + grad_c/norm_grad
            delta.grad.zero_()
            delta.data = delta.data - lr *torch.sign(g)
            delta.data = delta.data.clamp(-eps/127 ,eps/127)
            delta.data = ((batch + delta.data)).clamp(-1,1) - batch
            delta.data = delta.data
        return delta.data.mul_(0.5).mul_(255)

class contrastive_opposite(contrastive_attack):
    def __init__(self,retina,device,nets):
        super().__init__(retina,device)
        self.nets = nets
        self.sg = std_gan(device)
        self.ce = nn.CrossEntropyLoss().to(self.device)

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

    def flow_cont(self,b_t,net,ort,ori_feat,aug_feat,batch_size,is_ada,ll_idx):
        if(is_ada):
            ada_bt = self.rgb2bgr(b_t)
            feat,_ = net(ada_bt)
        else:
            feat = net(b_t)
        target = torch.cat((ort,ori_feat,aug_feat),dim=0)
        feat = torch.cat([feat[:batch_size*2],feat[batch_size*2:]],dim=1)
        feat = F.normalize(feat)
        loss = (self.cont_loss(feat[:batch_size],target)+self.cont_loss(feat[batch_size:],target))/2
        loss2 = F.cosine_similarity(feat[:batch_size],feat[batch_size:],dim=1).mean()
        loss3 = ((torch.var(feat[:batch_size],dim=1) - torch.var(ort,dim=1))**2).mean()
        loss4 = ((torch.mean(feat[:batch_size],dim=1)-torch.mean(ort,dim=1))**2).mean()

        return loss - loss2*0.001 + loss3 + loss4
 

    def attack(self,batch,batch_size=32,img_size=256,iter=500,lr=0.001,eps=16):
        device = self.device
        retinaface = self.retinaface.to(device)
        sg = self.sg
        batch = batch.to(device)
        new_shape = int(112)
        nets = self.nets.to(device)
        ran_crop = torchvision.transforms.RandomResizedCrop((img_size,img_size),scale=(0.8,0.9))
        g = torch.zeros_like(batch, requires_grad=True).to(device)
        with torch.no_grad():
            b_boxs,b_lnds = face_detection(batch,retinaface,device)
            augmented = sg.augment_full(batch).to(device)
        blob,angle = self.make_blob(batch,b_boxs,b_lnds)
        blob = blob.div_(255).sub_(0.5).div_(0.5)
        feat_list,ort_list = [],[]
        for key in nets.keys():
            if(key=='ada'):
                ada_blob = self.rgb2bgr(blob)
                feat,_ = nets[key](ada_blob)
            else: feat = nets[key](blob)
            feat = torch.cat([feat[:batch_size],feat[batch_size:]],dim=1)
            feat_list.append(feat.detach().clone())
        ll_idx = self.find_ll(feat_list[0])
        for feat in feat_list:
            ort = -1*feat+(torch.randn(feat.shape).to(device)/feat.sum())/feat.shape[0]
            ort = F.normalize(ort)
            feat = F.normalize(feat)
            ort_list.append(ort.detach().clone())
        del blob,ada_blob,feat,ort
        aug_blob,_ = self.make_blob(augmented,torch.cat((b_boxs,b_boxs,b_boxs,b_boxs),dim=0),angle=np.concatenate((angle,angle,angle,angle),axis=0))
        aug_blob = aug_blob.div_(255).sub_(0.5).div_(0.5)
        aug_feats = []
        for key in nets.keys():
            if(key=='ada'):
                ada_blob = self.rgb2bgr(aug_blob)
                feat,_ = nets[key](ada_blob)
            else: feat = nets[key](aug_blob)
            feat = torch.cat([feat[:batch_size*4],feat[batch_size*4:]],dim=1)
            feat = F.normalize(feat)
            aug_feats.append(feat.detach().clone())
        del aug_blob,ada_blob,feat,augmented
        batch = batch.div_(255).sub_(0.5).div_(0.5)
        tdelta = torch.zeros_like(batch).to(device)
        idx = 0
        for key in nets.keys():
            if(key=='ada'):
                tdelta=self.single_cont(nets[key],batch+tdelta,batch_size,ort_list[idx],feat_list[idx],aug_feats[idx],angle,b_boxs,ll_idx,iter//2,True,eps=eps/2)
                idx+=1
            else:
                tdelta=self.single_cont(nets[key],batch+tdelta,batch_size,ort_list[idx],feat_list[idx],aug_feats[idx],angle,b_boxs,ll_idx,iter//2,eps=eps/2)
                idx+=1
        delta = torch.zeros_like(batch).to(device)+tdelta.detach().clone()
        torch.cuda.empty_cache()
        delta.requires_grad_()
        for k in range (iter) :
            b_crop = ran_crop(batch)
            b_crop_mul = b_crop.detach().clone()
            c_boxs,_ = face_detection(b_crop_mul.mul_(0.5).add_(0.5).mul_(255),retinaface,device)
            b_t = torch.cat([batch + delta,b_crop + delta],dim=0)
            b_t,_ = self.make_blob(b_t,torch.cat((b_boxs,c_boxs),dim=0),angle=np.concatenate((angle,angle),axis=0))
            p = torch.randint(low = int(new_shape * 0.8), high = new_shape,size = (1,))[0]
            b_t = torchvision.transforms.functional.resize(b_t,(p,p))
            rest = new_shape-p
            top = np.random.randint(low= 0, high=rest)
            left = np.random.randint(low= 0, high=rest)
            b_t= F.pad(b_t,(left,rest-left,top,rest-top),mode='constant',value=0)
            loss = 0
            idx = 0
            loss = torch.zeros(len(nets))
            for key in nets.keys():
                if(key =='ada'):
                    loss[idx]=self.flow_cont(b_t,nets[key],ort_list[idx],feat_list[idx],aug_feats[idx],batch_size,True,ll_idx)
                    idx+=1
                else:
                    loss[idx]=self.flow_cont(b_t,nets[key],ort_list[idx],feat_list[idx],aug_feats[idx],batch_size,False,ll_idx)
                    idx+=1
            constraint = self.mse_loss(batch,batch+delta)
            out = loss.mean() + 0.1*constraint
            print(out)
            out.backward()
            grad_c = delta.grad.clone()
            norm_grad = torch.norm(grad_c,p=1)
            g = g + grad_c/norm_grad
            delta.grad.zero_()
            delta.data = delta.data - lr *torch.sign(g)
            delta.data = delta.data.clamp(-eps/127 ,eps/127)
            delta.data = ((batch + delta.data)).clamp(-1,1) - batch
            delta.data = delta.data
        return delta.data.mul_(0.5).mul_(255)


class margin_attack(Base):
    def __init__(self,retina,device,nets):
        super().__init__(retina,device)
        self.nets = nets
    def attack(self,batch,batch_size=32,img_size=256,iter=500,lr=0.001,eps=16):
        device = self.device
        retinaface = self.retinaface.to(device)
        batch = batch.to(device)
        nets = self.nets.to(device)
        g = torch.zeros_like(batch, requires_grad=True).to(device)
        with torch.no_grad():
            b_boxs,b_lnds = face_detection(batch,retinaface,device)
        blob,angle = self.make_blob(batch,b_boxs,b_lnds)
        blob = blob.div_(255).sub_(0.5).div_(0.5)
        feat_list,ort_list = [],[]
        for key in nets.keys():
            if(key=='ada'):
                ada_blob = self.rgb2bgr(blob)
                feat,_ = nets[key](ada_blob)
            else: feat = nets[key](blob)
            feat = torch.cat([feat[:batch_size],feat[batch_size:]],dim=1)
            feat_list.append(feat.detach().clone())
        ll_idx = self.find_ll(feat_list[0])
        for feat in feat_list:
            ort = self.permute_victim(feat,ll_idx)
            ort = F.normalize(ort)
            ort_list.append(ort.detach().clone())
        idx = 0
        del feat_list
        batch = batch.div_(255).sub_(0.5).div_(0.5)
        tdelta = torch.zeros_like(batch).to(device)
        for key in nets.keys():
            if(key=='ada'):
                tdelta=self.margin_single_attack(nets[key],batch+tdelta,batch_size,ort_list[idx],angle,b_boxs,ll_idx,iter//2,True,eps=eps/2)
                idx+=1
            else:
                tdelta=self.margin_single_attack(nets[key],batch+tdelta,batch_size,ort_list[idx],angle,b_boxs,ll_idx,iter//2,eps=eps/2)
                idx+=1
        delta = torch.zeros_like(batch).to(device)+tdelta
        delta.requires_grad_()
        new_shape = int(112)
        ran_crop = torchvision.transforms.RandomResizedCrop((img_size,img_size),scale=(0.85,0.95))
        for k in range (iter) :
            b_crop = ran_crop(batch)
            b_crop_mul = b_crop.detach().clone()
            c_boxs,_ = face_detection(b_crop_mul.mul_(0.5).add_(0.5).mul_(255),retinaface,device)
            b_t = torch.cat([batch + delta,b_crop + delta],dim=0)
            box_a = torch.cat((b_boxs,c_boxs),dim=0)
            b_t,_ = self.make_blob(b_t,box_a,angle=np.concatenate((angle,angle),axis=0))
            p = torch.randint(low = int(new_shape * 0.8), high = new_shape,size = (1,))[0]
            b_t = torchvision.transforms.functional.resize(b_t,(p,p))
            rest = new_shape-p
            top = np.random.randint(low= 0, high=rest)
            left = np.random.randint(low= 0, high=rest)
            b_t= F.pad(b_t,(left,rest-left,top,rest-top),mode='constant',value=0)
            losses=torch.zeros(len(nets))
            idx=0
            for key in nets.keys():
                if(key =='ada'):
                    losses[idx]=self.flow_margin(b_t,nets[key],ort_list[idx],batch_size,True,ll_idx)
                    idx+=1
                else:
                    losses[idx]=self.flow_margin(b_t,nets[key],ort_list[idx],batch_size,False,ll_idx)
                    idx+=1
            loss = losses.mean()
            constraint = self.mse_loss(batch,batch+delta)
            out = loss + 0.05*constraint
            print(out)
            out.backward()
            grad_c = delta.grad.clone()
            norm_grad = torch.norm(grad_c,p=1)
            g = g + grad_c/norm_grad
            delta.grad.zero_()
            delta.data = delta.data - lr *torch.sign(g)
            delta.data = delta.data.clamp(-eps/127 ,eps/127)
            delta.data = ((batch + delta.data)).clamp(-1,1) - batch
        return delta.data.mul_(0.5).mul_(255)


