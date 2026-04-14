import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import os

def torch_to_cv(batch):
  imgs = batch.detach().cpu().numpy()
  imgs = np.transpose(imgs,(0,2,3,1))
  imgs = np.clip(imgs,0,255).astype(np.uint8)
  imgs = [cv.cvtColor(img,cv.COLOR_RGB2BGR) for img in imgs]
  return imgs

def torch_to_pil(batch):
  imgs = batch.detach().cpu().numpy()
  imgs = np.transpose(imgs,(0,2,3,1))
  imgs = np.uint8(imgs)
  return imgs

def save_multi(imgs,start_index,batch_num,dirname):
  cv_imgs = torch_to_cv(imgs)
  for i in range(batch_num):
    j=start_index*batch_num 
    cv.imwrite(os.path.join(dirname,str(j+i)+'.jpg'),cv_imgs[i])

def make_orthogonal(u):
  v = np.random.normal(u.min(),u.max(),u.shape)
  v = v-((u.dot(v)/u.dot(u))*u)
  return v

def resize_to_recognition(img, size=(112,112)):
  return F.interpolate(img, size=size, mode='bilinear', align_corners=True)

def random_resize_crop(img, size=(112,112), scale_range=0.1, shift_range=2.0):
  batch_size = img.shape[0]
  device = img.device
  scale = 1.0 + (torch.rand(batch_size, 1, 1, device=device) - 0.5) * 2 * scale_range
  shift = (torch.rand(batch_size, 2, 1, device=device) - 0.5) * (2 * shift_range / size[0] * 2)
  theta = torch.zeros(batch_size, 2, 3, device=device)
  theta[:, 0, 0] = scale.squeeze()
  theta[:, 1, 1] = scale.squeeze()
  theta[:, :, 2:] = shift
  grid = F.affine_grid(theta, torch.Size((batch_size, img.shape[1], size[0], size[1])), align_corners=True)
  return F.grid_sample(img, grid, mode='bilinear', align_corners=True)

