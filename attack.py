from util import resize_to_recognition, random_resize_crop, differentiable_warping
import os
import sys
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import lpips
import gc
from matplotlib import pyplot as plt
try:
    from pytorch_wavelets import DWTForward
    _DWT_OK = True
except Exception:
    _DWT_OK = False


_RF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RetinaFace_Pytorch')
if _RF_PATH not in sys.path:
    sys.path.insert(0, _RF_PATH)


def face_detection(batch_m11, retinaface, device):
    """RetinaFace 5-point detection. Input `batch_m11` is a float tensor in
    [-1,1] RGB, [B,3,H,W]. Falls back to a canonical default landmark layout
    when detection fails so the warp downstream stays well-defined.
    Ported from old_attack.py (kept deliberately close)."""
    import eval_widerface
    boxes, landmarks = [], []
    batch = (batch_m11 * 0.5 + 0.5) * 255.
    img_h, img_w = batch.shape[2], batch.shape[3]
    with torch.no_grad():
        picked_boxes, picked_landmarks, picked_scores = eval_widerface.get_detections(
            batch, retinaface, score_threshold=0.5, iou_threshold=0.3)
        for b, s, l in zip(picked_boxes, picked_scores, picked_landmarks):
            if s is None:
                box = torch.tensor([0, 0, img_w, img_h], dtype=torch.float32, device=device)
                landmark = torch.tensor([
                    [img_w * 0.3, img_h * 0.3],
                    [img_w * 0.7, img_h * 0.3],
                    [img_w * 0.5, img_h * 0.5],
                    [img_w * 0.3, img_h * 0.7],
                    [img_w * 0.7, img_h * 0.7],
                ], dtype=torch.float32, device=device)
                boxes.append(box)
                landmarks.append(landmark)
            else:
                ss = torch.squeeze(s, dim=0)
                bb = torch.squeeze(b, dim=0)
                if ss.shape[0] > 1:
                    boxes.append(bb[torch.argmax(ss)].squeeze(0))
                    landmarks.append(l[torch.argmax(ss)].squeeze(0).view(5, 2))
                else:
                    boxes.append(bb.squeeze(0))
                    landmarks.append(l.squeeze(0).view(5, 2))
    return torch.stack(boxes, dim=0), torch.stack(landmarks, dim=0)


class contrastive_opposite(nn.Module):
    def __init__(self,device,nets,lpips_weight=0.1,retinaface=None,loss_type='ce',grad_smooth='none',regularizer='lpips',reg_weight=None):
        super(contrastive_opposite,self).__init__()
        self.mse_loss = nn.MSELoss()
        self.csloss = nn.CosineEmbeddingLoss()
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.nets = nets.to(device)
        self.ce = nn.CrossEntropyLoss().to(self.device)
        self.kernel = self.gkern(klen=5, nsig=1.0)
        self.retinaface = retinaface
        self.loss_type = loss_type
        self.grad_smooth = grad_smooth
        # Regularizer selection: 'lpips' | 'wavelet' | 'none'.
        self.regularizer = regularizer
        # Keep lpips_weight for backwards compat; reg_weight overrides when provided.
        self.lpips_weight = lpips_weight
        self.reg_weight = reg_weight if reg_weight is not None else lpips_weight
        self.lpips = None
        self.dwt = None
        self._ssim_loss_fn = None
        if self.regularizer == 'lpips':
            self.lpips = lpips.LPIPS(net='vgg').to(device).eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)
        elif self.regularizer == 'wavelet':
            assert _DWT_OK, "pytorch_wavelets not installed; pip install pytorch_wavelets PyWavelets"
            self.dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)
            for p in self.dwt.parameters():
                p.requires_grad_(False)
        elif self.regularizer == 'ssim':
            from kornia.losses import ssim_loss as _kornia_ssim_loss
            self._ssim_loss_fn = _kornia_ssim_loss
        elif self.regularizer != 'none':
            raise ValueError(f'unknown regularizer: {self.regularizer}')

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
        bgrims = torch.flip(imgs,[1]).contiguous()
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

    def flow_cont(self,b_t,net,ort,ori_feat,ori_aug_feat,aug_feat,batch_size,is_ada):
        if(is_ada):
            ada_bt = self.rgb2bgr(b_t)
            feat,_ = net(ada_bt)
        else:
            feat = net(b_t)
        feat = F.normalize(feat)
        target = torch.cat((ort,ori_feat,ori_aug_feat,aug_feat),dim=0)
        loss = self.cont_loss(feat,target)
        return loss,feat

    def supcon_loss(self, feat, targets, pos_mask, tau=0.07):
        sim = F.linear(feat, targets) / tau
        log_prob = sim - sim.logsumexp(dim=1, keepdim=True)
        loss = -(pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)
        return loss.mean()

    def flow_supcon(self, b_t, net, pos_targets, neg_targets, batch_size, is_ada):
        """SupCon loss with arbitrary positive/negative target lists.
        pos_targets: [N_pos, D], neg_targets: [N_neg, D].
        Both are laid out as consecutive B-sized blocks where block k's
        sample i corresponds to anchor i."""
        if is_ada:
            ada_bt = self.rgb2bgr(b_t)
            feat, _ = net(ada_bt)
        else:
            feat = net(b_t)
        feat = F.normalize(feat)
        B = batch_size
        targets = torch.cat([pos_targets, neg_targets], dim=0)
        n_pos_blocks = pos_targets.shape[0] // B
        N = targets.shape[0]
        pos_mask = torch.zeros(B, N, device=self.device)
        arange_B = torch.arange(B, device=self.device)
        for blk in range(n_pos_blocks):
            pos_mask[arange_B, blk * B + arange_B] = 1.0
        loss = self.supcon_loss(feat, targets, pos_mask)
        return loss, feat

    def _encode(self,blob_m11,key):
        if(key=='ada'):
            ada_blob = self.rgb2bgr(blob_m11)
            feat,_ = self.nets[key](ada_blob)
        else:
            feat = self.nets[key](blob_m11)
        return F.normalize(feat)

    def ready_made_attack(self,batch):
        blob = resize_to_recognition(batch)
        blob = blob.div(255).sub(0.5).div(0.5)
        feat_list,ort_list = [],[]
        for key in self.nets.keys():
            feat = self._encode(blob,key).detach().clone()
            feat_list.append(feat)
            ort_list.append(F.normalize(-1*feat).detach().clone())
        return ort_list,feat_list

    def _ready_aligned(self, batch_o, pair_o):
        """Aligned version of ready_made_attack. Runs RetinaFace once on the
        source and once on the pair, caches their affines, and builds the
        per-recognizer (ort, ori_feat) targets from an aligned 112x112 crop.
        Returns (ort_list, feat_list, src_affines, src_aligned_255, pair_aligned_255)."""
        batch_m11 = (batch_o / 255. - 0.5) / 0.5
        pair_m11 = (pair_o / 255. - 0.5) / 0.5
        with torch.no_grad():
            b_boxes, b_lnds = face_detection(batch_m11, self.retinaface, self.device)
            p_boxes, p_lnds = face_detection(pair_m11, self.retinaface, self.device)
            src_warped_255, src_affines = differentiable_warping(
                self.device, batch_o, b_boxs=b_boxes, b_lnds=b_lnds,
                affine_arrays=None, mi_transform=False, target_shape=(112, 112))
            pair_warped_255, _ = differentiable_warping(
                self.device, pair_o, b_boxs=p_boxes, b_lnds=p_lnds,
                affine_arrays=None, mi_transform=False, target_shape=(112, 112))
        src_blob = (src_warped_255 / 255. - 0.5) / 0.5
        feat_list, ort_list = [], []
        for key in self.nets.keys():
            feat = self._encode(src_blob, key).detach().clone()
            feat_list.append(feat)
            ort_list.append(F.normalize(-1 * feat).detach().clone())
        return ort_list, feat_list, src_affines, src_warped_255.detach(), pair_warped_255.detach()

    def _encode_pair_views(self,pair_m11):
        pair_rrc = random_resize_crop(pair_m11)
        views = torch.cat([pair_m11, pair_m11.fliplr(),
                           pair_rrc, pair_rrc.fliplr()], dim=0)
        feats_per_net = []
        for key in self.nets.keys():
            feats_per_net.append(self._encode(views,key).detach().clone())
        return feats_per_net

    def _encode_pair_views_aligned(self, pair_aligned_255):
        """Pair views for the aligned path. pair_aligned_255 is already a
        112x112 aligned crop in [0,255]. We normalize to [-1,1] and then
        reuse the same plain/flip/RRC/RRC+flip pool."""
        pair_m11 = (pair_aligned_255 / 255. - 0.5) / 0.5
        return self._encode_pair_views(pair_m11)

    def _encode_multi_aug(self, images_m11_list):
        """Encode augmented views (RRC + flipped RRC) for a list of images.
        Each image [B,3,H,W] produces 4B features (plain/flip/RRC/RRC+flip).
        Returns per-net list of [total_views, D] tensors.
        A fresh RRC is drawn per call so pos/neg get different crops."""
        all_views = []
        for img in images_m11_list:
            rrc = random_resize_crop(img)
            all_views.append(torch.cat([img, img.fliplr(), rrc, rrc.fliplr()], dim=0))
        combined = torch.cat(all_views, dim=0)
        feats_per_net = []
        for key in self.nets.keys():
            feats_per_net.append(self._encode(combined, key).detach().clone())
        return feats_per_net

    def attack(self,batch_o,pair_o,batch_size=32,img_size=256,iter=500,lr=0.001,eps=16,use_alignment=False):
        """pair_o: either [B,3,H,W] (single pair) or list of [B,3,H,W] (multi-pair)."""
        batch_o = batch_o.to(self.device)
        if isinstance(pair_o, (list, tuple)):
            pairs_o = [p.to(self.device) for p in pair_o]
        else:
            pairs_o = [pair_o.to(self.device)]
        n_pairs = len(pairs_o)

        if use_alignment:
            assert self.retinaface is not None, "use_alignment=True requires retinaface in __init__"
            ort_list, feat_list, src_affines, src_aligned_255, pair_aligned_255 = self._ready_aligned(batch_o, pairs_o[0])
            pairs_aligned_255 = [pair_aligned_255]
            for p in pairs_o[1:]:
                p_m11 = (p / 255. - 0.5) / 0.5
                with torch.no_grad():
                    pb, pl = face_detection(p_m11, self.retinaface, self.device)
                    pw, _ = differentiable_warping(self.device, p, b_boxs=pb, b_lnds=pl,
                                                   affine_arrays=None, mi_transform=False, target_shape=(112, 112))
                pairs_aligned_255.append(pw.detach())
        else:
            ort_list, feat_list = self.ready_made_attack(batch_o)
            src_affines = None
            src_aligned_255 = None
            pairs_aligned_255 = None
        feat_list = torch.stack(feat_list, dim=0)
        batch = ((batch_o / 255. - 0.5) / 0.5)
        source_small = resize_to_recognition(batch)
        pairs_small = [resize_to_recognition((p / 255. - 0.5) / 0.5) for p in pairs_o]

        # Pre-compute plain pair features (for supcon positive ort_pair / negative pair_feat)
        pairs_feat_list = []  # per-net list of [n_pairs*B, D]
        with torch.no_grad():
            for key in self.nets.keys():
                pfeats = []
                for pi in range(n_pairs):
                    if use_alignment:
                        pb = (pairs_aligned_255[pi] / 255. - 0.5) / 0.5
                    else:
                        pb = pairs_small[pi]
                    pfeats.append(self._encode(pb, key).detach().clone())
                pairs_feat_list.append(torch.cat(pfeats, dim=0))

        g = torch.zeros_like(batch).to(self.device)
        delta = torch.zeros_like(batch, requires_grad=True).to(self.device)
        sims = np.zeros((len(self.nets), iter))
        torch.cuda.empty_cache()
        for k in range(iter):
            idx = 0
            b_t = batch + delta
            if use_alignment:
                b_t_255 = (b_t * 0.5 + 0.5) * 255.
                b_t_warped_255, _ = differentiable_warping(
                    self.device, b_t_255, affine_arrays=src_affines,
                    mi_transform=True, target_shape=(112, 112))
                b_t_small = (b_t_warped_255 / 255. - 0.5) / 0.5
            else:
                b_t_small = resize_to_recognition(b_t)
            with torch.no_grad():
                if use_alignment:
                    all_imgs = [src_aligned_255] + pairs_aligned_255
                    all_imgs_m11 = [(x / 255. - 0.5) / 0.5 for x in all_imgs]
                else:
                    all_imgs_m11 = [source_small] + pairs_small
                neg_aug_feats_k = self._encode_multi_aug(all_imgs_m11)  # 20B per net
                if self.loss_type == 'supcon':
                    pos_aug_feats_k = self._encode_multi_aug(all_imgs_m11)  # separate RRC draw
            losses = []
            new_feats = torch.zeros_like(feat_list)
            for key in self.nets.keys():
                is_ada = (key == 'ada')
                if self.loss_type == 'supcon':
                    # Positives: ort(B) + ort_pairs(n_pairs*B) + inverted_aug(20B) = (1+n_pairs+20)B = 25B
                    pos_targets = torch.cat([
                        ort_list[idx],
                        F.normalize(-1 * pairs_feat_list[idx]),
                        F.normalize(-1 * pos_aug_feats_k[idx]),
                    ], dim=0)
                    # Negatives: ori_feat(B) + pairs_feat(n_pairs*B) + aug(20B) = 25B
                    neg_targets = torch.cat([
                        feat_list[idx],
                        pairs_feat_list[idx],
                        neg_aug_feats_k[idx],
                    ], dim=0)
                    l, new_feats[idx] = self.flow_supcon(
                        b_t_small, self.nets[key], pos_targets, neg_targets, batch_size, is_ada)
                else:
                    # CE path: ori_feat(B) neg + ori_aug(from source, first 4B of neg_aug) + pair_aug(rest)
                    ori_aug = neg_aug_feats_k[idx][:4*batch_size]
                    pair_aug = neg_aug_feats_k[idx][4*batch_size:]
                    all_neg_aug = torch.cat([ori_aug, pairs_feat_list[idx], pair_aug], dim=0)
                    l, new_feats[idx] = self.flow_cont(
                        b_t_small, self.nets[key],
                        ort_list[idx], feat_list[idx],
                        ori_aug, torch.cat([pairs_feat_list[idx], pair_aug], dim=0),
                        batch_size, is_ada)
                losses.append(l)
                idx += 1
            out = torch.stack(losses).mean()
            if self.reg_weight > 0 and self.regularizer != 'none':
                if self.regularizer == 'lpips':
                    reg = self.lpips(batch, batch + delta).mean()
                elif self.regularizer == 'wavelet':
                    LL, _ = self.dwt(delta)
                    reg = (LL ** 2).mean()
                else:  # 'ssim' — kornia ssim_loss returns (1 - SSIM); inputs in [0, 1]
                    reg = self._ssim_loss_fn(
                        (batch + 1) / 2, (batch + delta + 1) / 2, window_size=11)
                out = out + self.reg_weight * reg
            out.backward()
            grad_c = delta.grad.clone()
            if self.grad_smooth == 'ti':
                grad_c = F.conv2d(grad_c,self.kernel,padding=5//2,groups=3)
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
        self.last_sims = sims  # [n_nets, iter] cosine-to-original trajectory (for plotting/analysis)
        return delta.data * 127.5

class direct_attack(contrastive_opposite):
    def __init__(self,device,nets,lpips_weight=0.1):
        super(direct_attack,self).__init__(device,nets,lpips_weight=lpips_weight)

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
            if self.reg_weight > 0 and self.regularizer != 'none':
                if self.regularizer == 'lpips':
                    reg = self.lpips(batch, batch + delta).mean()
                elif self.regularizer == 'wavelet':
                    LL, _ = self.dwt(delta)
                    reg = (LL ** 2).mean()
                else:  # 'ssim'
                    reg = self._ssim_loss_fn(
                        (batch + 1) / 2, (batch + delta + 1) / 2, window_size=11)
                out = out + self.reg_weight * reg
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