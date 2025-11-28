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

def get_affine(b_boxs,b_lnds=None,angle=None,target_size=(112,112)):
  x1, y1, x2, y2 = b_boxs
  w_box = x2 - x1
  h_box = y2 - y1
  cx, cy = w_box / 2.0, h_box / 2.0 # Box 내부의 중심 좌표

  # 1. M_crop: 원본 좌표계에서 Box 좌상단을 원점으로 이동
  # [1, 0, -x1]
  # [0, 1, -y1]
  # [0, 0,  1]
  M_crop = np.eye(3)
  M_crop[0, 2] = -x1
  M_crop[1, 2] = -y1

  # 2. M_rot: Box 중심 기준으로 회전
  # (1) 중심으로 이동 -> (2) 회전 -> (3) 다시 복구
  
  # 각도 변환 (Degree -> Radian)
  # Torchvision rotate는 시계 반대 방향(CCW)이 양수, 
  # 좌표계 변환에서는(y축이 아래) 반대로 작용할 수 있으나, 
  # 일반적인 2D 회전 행렬을 사용하고 방향을 맞춥니다.
  # angle이 양수일 때 CCW 회전이라 가정:
  if angle is None:
      angle_deg = get_rot_angle(b_lnds[0],b_lnds[1])
  else:
      angle_deg = angle
  theta = np.radians(angle_deg)
  cos_a = np.cos(theta)
  sin_a = np.sin(theta)

  # 회전 행렬 (CCW)
  R = np.array([
      [cos_a, -sin_a, 0],
      [sin_a,  cos_a, 0],
      [0,      0,     1]
  ])

  # 중심 이동 행렬 T
  T_to_center = np.eye(3)
  T_to_center[0, 2] = -cx
  T_to_center[1, 2] = -cy

  T_from_center = np.eye(3)
  T_from_center[0, 2] = cx
  T_from_center[1, 2] = cy

  # 결합: 중심기준 회전 = T_inv * R * T
  M_rot_step = T_from_center @ R @ T_to_center

  # 3. M_scale: (w_box, h_box) -> (112, 112) 리사이징
  M_scale = np.eye(3)
  M_scale[0, 0] = target_size[0] / w_box
  M_scale[1, 1] = target_size[1] / h_box

  # 4. 최종 행렬 결합 (순서: 뒤에서 앞으로 적용되므로 행렬 곱은 왼쪽에서 오른쪽으로)
  # Target = Scale * (Rot_at_Center) * Crop * Source
  M_total = M_scale @ M_rot_step @ M_crop

  return M_total[:2, :] # 2x3 행렬 반환

def get_torch_affine(M, src_shape,target_shape):
  """
  OpenCV의 Affine Matrix (2x3)를 PyTorch의 affine_grid용 Theta (N, 2, 3)로 변환합니다.
  PyTorch는 [-1, 1]의 Normalized Coordinate를 사용하므로 이에 맞춘 스케일링이 필요합니다.
  """
  H, W = src_shape
  
  # 1. OpenCV Matrix 확장을 통해 3x3으로 만듦 (Homogeneous coordinates)
  M_aug = np.vstack([M, [0, 0, 1]])
  
  # 2. Inverse 계산
  # grid_sample은 Target 픽셀이 Source의 어디에 해당하는지를 찾으므로(Inverse mapping),
  # 보통 Forward Matrix의 역행렬이 필요합니다.
  M_inv = np.linalg.inv(M_aug)
  
  # 3. Normalize 좌표계 변환 행렬
  # [0, W] -> [-1, 1] 로 가는 변환 행렬들
  S_src = np.array([[2/W, 0, -1], [0, 2/H, -1], [0, 0, 1]])
  S_dst = np.array([[2/target_shape[0], 0, -1], [0, 2/target_shape[1], -1], [0, 0, 1]])

  # 최종 변환: Theta = S_src * M_inv * S_dst^(-1)
  # (Target 정규좌표 -> Target 픽셀 -> Source 픽셀 -> Source 정규좌표)
  M_torch = S_src @ M_inv @ np.linalg.inv(S_dst)
  
  # PyTorch는 2x3 행렬을 요구 (마지막 행 제거)
  return torch.from_numpy(M_torch[:2, :]).float()


def mi_on_affine(affine,batch_size,device):
  scale_noise = 1.0 + (torch.rand(batch_size, 1, 1, device=device) - 0.5) * 0.2 
  # Affine Matrix의 앞부분(회전/크기)에 곱해줌
  M_noisy = affine.clone()
  M_noisy[:, :, :2] = affine[:, :, :2] * scale_noise

  # (B) Translation Noise (Shift)
  # -2 ~ +2 픽셀 정도 이동
  shift_noise = (torch.rand(batch_size, 2, 1, device=device) - 0.5) * (4.0 / 112.0 * 2.0) 
  M_noisy[:, :, 2:] = affine[:, :, 2:] + shift_noise
  return M_noisy


def get_grid(device,b_boxs,b_lnds=None,angle=None,src_shape=(112,112),target_shape=(112,112),batch_size=1,affine_arrays=None,mi_transform=True):
  theta_batch = torch.zeros((batch_size,2,3),device=device) if affine_arrays is None else affine_arrays
  if affine_arrays is None:
    for b in range(batch_size):
        M = get_affine(b_boxs[b],b_lnds=b_lnds[b],angle=angle) 
        theta = get_torch_affine(M,src_shape,target_shape).to(device)
        theta_batch[b] = theta.clone()
  noisy_theta = mi_on_affine(theta_batch,batch_size,device) if mi_transform else theta_batch
  grid = F.affine_grid(noisy_theta,torch.Size((batch_size,3,target_shape[1],target_shape[0])),align_corners=True)
  return grid,theta_batch

def differentiable_warping(device,img, b_boxs,b_lnds=None,angle=None,affine_arrays=None,mi_transform=True):
  batch_size = img.shape[0]
  grid,affines = get_grid(device,b_boxs,b_lnds=b_lnds,angle=angle,src_shape=(img.shape[3],img.shape[2]),
                          target_shape=(112,112),batch_size=batch_size,affine_arrays=affine_arrays,mi_transform=mi_transform)
  warped = F.grid_sample(img,grid,mode='bilinear',align_corners=True)
  return warped,affines

