from util import resize_to_recognition, random_resize_crop
import os
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from augmen import std_gan
import gc
from matplotlib import pyplot as plt


class contrastive_opposite(nn.Module):
    def __init__(self,device,nets):
        super(contrastive_opposite,self).__init__()
        self.mse_loss = nn.MSELoss()
        self.csloss = nn.CosineEmbeddingLoss()
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.nets = nets.to(device)
        self.sg = std_gan(device)
        self.ce = nn.CrossEntropyLoss().to(self.device)
        self.kernel = self.gkern(klen=5, nsig=1.0)

    def gkern(self,klen=5, nsig=1.0):
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
        logit = logit *s
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
        return loss,feat

    def ready_made_attack(self,batch):
        with torch.no_grad():
            augmented = self.sg.augment_full(batch).to(self.device)
        num_aug = augmented.shape[0] // batch.shape[0]
        blob = resize_to_recognition(batch)
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
            ort = -1*feat
            ort = F.normalize(ort)
            feat = F.normalize(feat)
            ort_list.append(ort.detach().clone())
        aug_blob = resize_to_recognition(augmented)
        aug_blob=torch.cat([aug_blob,aug_blob.fliplr()],dim=0)
        aug_blob = aug_blob.div_(255).sub_(0.5).div_(0.5)
        aug_feats = []
        for key in self.nets.keys():
            if(key=='ada'):
                ada_blob = self.rgb2bgr(aug_blob)
                feat,_ = self.nets[key](ada_blob)
            else: feat = self.nets[key](aug_blob)
            feat = torch.cat([feat[:batch.shape[0]*num_aug],feat[batch.shape[0]*num_aug:]],dim=1)
            feat = F.normalize(feat)
            aug_feats.append(feat.detach().clone())

        return ort_list,feat_list,aug_feats


    def attack(self,batch_o,batch_size=32,img_size=256,iter=500,lr=0.001,eps=16):
        ort_list,feat_list,aug_feats = self.ready_made_attack(batch_o.to(self.device))
        feat_list = torch.stack(feat_list,dim=0)
        batch = ((batch_o/255. - 0.5) /0.5).to(self.device)
        g = torch.zeros_like(batch).to(self.device)
        delta = torch.zeros_like(batch,requires_grad=True).to(self.device)
        sims = np.zeros((len(self.nets),iter))
        torch.cuda.empty_cache()
        for k in range (iter) :
            idx = 0
            b_t = batch + delta
            b_t = random_resize_crop(b_t)
            b_t = torch.cat([b_t,b_t.fliplr()],dim=0)
            losses = []
            new_feats = torch.zeros_like(feat_list)
            for key in self.nets.keys():
                if(key =='ada'):
                    l,new_feats[idx]=self.flow_cont(b_t,self.nets[key],ort_list[idx],feat_list[idx],aug_feats[idx],batch_size,True)
                else:
                    l,new_feats[idx]=self.flow_cont(b_t,self.nets[key],ort_list[idx],feat_list[idx],aug_feats[idx],batch_size,False)
                losses.append(l)
                idx+=1
            out = torch.stack(losses).mean()
            #print(out)
            out.backward()
            grad_c = delta.grad.clone()
            #grad_c = F.conv2d(grad_c,self.kernel,padding=5//2,groups=3)
            norm_grad = torch.norm(grad_c,p=1,dim=(1,2,3),keepdim=True)
            g = g + grad_c/norm_grad
            delta.grad.zero_()
            delta.data = delta.data - lr *torch.sign(g)
            delta.data = delta.data.clamp(-eps/127 ,eps/127)
            delta.data = ((batch + delta.data)).clamp(-1,1) - batch
            delta.data = delta.data
            with torch.no_grad():
                current_sim = F.cosine_similarity(new_feats, feat_list, dim=2)
                current_sim = current_sim.mean(dim=1).cpu().numpy()
                sims[:,k] = current_sim
        plt.figure(figsize=(10, 6))
        x_axis = np.arange(sims.shape[1])
        
        # 모델 Key 가져오기
        model_keys = list(self.nets.keys())
        
        # 각 모델별로 선 그리기
        for i in range(sims.shape[0]):
            label_name = model_keys[i] if i < len(model_keys) else f"Model {i}"
            plt.plot(x_axis, sims[i], label=label_name)
        
        plt.title("Cosine Similarity Degradation during Attack")
        plt.xlabel("Iterations")
        plt.ylabel("Avg Cosine Similarity")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1)) # 범례가 그래프 가리지 않게 밖으로
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        temp = os.listdir('./')
        temp2 = [i for i in temp if i.startswith('cont_flow')]
        plt.savefig(f'./cont_flow_{len(temp2)}.png', dpi=300)
        plt.close() # 메모리 해제
        return delta.data * 127.5

class direct_attack(contrastive_opposite):
    def __init__(self,device,nets):
        super(direct_attack,self).__init__(device,nets)

    def ready_made_attack(self, batch):
        blob = resize_to_recognition(batch)
        blob=torch.cat([blob,blob.fliplr()],dim=0)
        blob.div_(255).sub_(0.5).div_(0.5)
        feat_list = []
        for key in self.nets.keys():
            if(key=='ada'):
                ada_blob = self.rgb2bgr(blob)
                feat,_ = self.nets[key](ada_blob)
            else: feat = self.nets[key](blob)
            feat = torch.cat([feat[:batch.shape[0]],feat[batch.shape[0]:]],dim=1)
            feat_list.append(feat.detach().clone())
        feats = torch.stack(feat_list,dim=0)
        feats_norm = F.normalize(feats,p=2,dim=2)
        sim_mat = torch.bmm(feats_norm,feats_norm.permute(0,2,1))
        targets = torch.argmin(sim_mat,dim=2)
        sorted_feats = feats[torch.arange(feats.shape[0]).unsqueeze(1),targets]

        return sorted_feats,feats
    
    def flow(self,b_t,net,target,is_ada):
        if(is_ada):
            ada_bt = self.rgb2bgr(b_t)
            feat,_ = net(ada_bt)
        else:
            feat = net(b_t)
        feat = torch.cat([feat[:b_t.shape[0]//2],feat[b_t.shape[0]//2:]],dim=1)
        loss = F.cosine_similarity(feat, target, dim=1).mean()
        loss = -1*loss
        return loss,feat
    
    def attack(self,batch_o,batch_size=32,img_size=256,iter=500,lr=0.001,eps=16):
        target,feats = self.ready_made_attack(batch_o.to(self.device))
        batch = ((batch_o/255. - 0.5) /0.5).to(self.device)
        g = torch.zeros_like(batch).to(self.device)
        delta = torch.zeros_like(batch,requires_grad=True).to(self.device)
        sims = np.zeros((len(self.nets),iter))
        torch.cuda.empty_cache()
        for k in range (iter) :
            idx = 0
            b_t = batch + delta
            b_t = random_resize_crop(b_t)
            b_t = torch.cat([b_t,b_t.fliplr()],dim=0)
            losses = []
            new_feats = torch.zeros_like(feats)
            for key in self.nets.keys():
                if(key =='ada'):
                    l,new_feats[idx]=self.flow(b_t,self.nets[key],target,True)
                else:
                    l,new_feats[idx]=self.flow(b_t,self.nets[key],target,False)
                losses.append(l)
                idx+=1
            out = torch.stack(losses).mean()
            #print(out)
            out.backward()
            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c,self.kernel,padding=5//2,groups=3)
            norm_grad = torch.norm(grad_c,p=1,dim=(1,2,3),keepdim=True)
            g = g + grad_c/norm_grad
            delta.grad.zero_()
            delta.data = delta.data - lr *torch.sign(g)
            delta.data = delta.data.clamp(-eps/127 ,eps/127)
            delta.data = ((batch + delta.data)).clamp(-1,1) - batch
            delta.data = delta.data
            with torch.no_grad():
                current_sim = F.cosine_similarity(new_feats, feats, dim=2)
                current_sim = current_sim.mean(dim=1).cpu().numpy()
                sims[:,k] = current_sim
        plt.figure(figsize=(10, 6))
        x_axis = np.arange(sims.shape[1])
        
        # 모델 Key 가져오기
        model_keys = list(self.nets.keys())
        
        # 각 모델별로 선 그리기
        for i in range(sims.shape[0]):
            label_name = model_keys[i] if i < len(model_keys) else f"Model {i}"
            plt.plot(x_axis, sims[i], label=label_name)
        
        plt.title("Cosine Similarity Degradation during Attack")
        plt.xlabel("Iterations")
        plt.ylabel("Avg Cosine Similarity")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1)) # 범례가 그래프 가리지 않게 밖으로
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        temp = os.listdir('./')
        temp2 = [i for i in temp if i.startswith('normal_flow')]
        plt.savefig(f'./normal_flow_{len(temp2)}.png', dpi=300)
        plt.close() # 메모리 해제

        return delta.data * 127.5