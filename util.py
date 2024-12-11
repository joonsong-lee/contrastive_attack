import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision
import math
import os

def torch_to_cv(batch):
  imgs = batch.detach().cpu().numpy()
  imgs = np.transpose(imgs,(0,2,3,1))
  imgs = np.uint8(imgs)
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

def angle_between_three_points(A, B, C):
  BA = [A[0]-B[0], A[1]-B[1]]
  BC = [C[0]-B[0], C[1]-B[1]]

  dot_product = BA[0]*BC[0] + BA[1]*BC[1]
  magnitude_BA = math.sqrt(BA[0]**2 + BA[1]**2)
  magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)

  cos_theta = dot_product / (magnitude_BA * magnitude_BC)

  if cos_theta > 1:
    cos_theta = 1
  elif cos_theta < -1:
    cos_theta = -1
  angle = math.acos(cos_theta)

  return math.degrees(angle)

def get_rot_angle(eye1,eye2):
  if(eye1[1]<eye2[1]):
    angle = angle_between_three_points(eye1,eye2,[eye1[0],eye2[1]])

  else :
    angle = angle_between_three_points(eye2,eye1,[eye2[0],eye1[1]]) *-1

  return angle


def make_orthogonal(u):
  v = np.random.normal(u.min(),u.max(),u.shape)
  v = v-((u.dot(v)/u.dot(u))*u)
  return v

