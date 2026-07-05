try:
    import ml_dtypes as _mld
    for _m in ('float4_e2m1fn', 'float6_e2m3fn', 'float6_e3m2fn', 'float8_e8m0fnu'):
        if not hasattr(_mld, _m):
            setattr(_mld, _m, _mld.float8_e4m3fn)
except Exception:
    pass

import cv2 as cv
import numpy as np
import math
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


# ---------------------------------------------------------------------------
# RetinaFace-driven differentiable similarity warp (ported from old_util.py).
# A face box + eye landmarks produce a cv2-style 2x3 affine (translate-to-crop,
# rotate around box center so eyes are level, scale to target_shape), which is
# then converted to an F.affine_grid-compatible theta in the normalized [-1,1]
# coordinate frame. grid_sample makes the whole stack differentiable so
# gradients from the recognizer flow back into the raw source pixels.
# ---------------------------------------------------------------------------

def angle_between_three_points(A, B, C):
  BA = [A[0]-B[0], A[1]-B[1]]
  BC = [C[0]-B[0], C[1]-B[1]]
  dot_product = BA[0]*BC[0] + BA[1]*BC[1]
  magnitude_BA = math.sqrt(BA[0]**2 + BA[1]**2)
  magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)
  denom = (magnitude_BA * magnitude_BC)
  if denom == 0:
    return 0.0
  cos_theta = dot_product / denom
  cos_theta = max(-1.0, min(1.0, cos_theta))
  return math.degrees(math.acos(cos_theta))


def get_rot_angle(eye1, eye2):
  if eye1[1] < eye2[1]:
    return angle_between_three_points(eye1, eye2, [eye1[0], eye2[1]])
  return angle_between_three_points(eye2, eye1, [eye2[0], eye1[1]]) * -1


def get_affine(b_box, b_lnds=None, angle=None, target_size=(112,112)):
  x1, y1, x2, y2 = [float(v) for v in b_box]
  w_box = x2 - x1
  h_box = y2 - y1
  if w_box <= 1 or h_box <= 1:
    w_box = max(w_box, 1.0)
    h_box = max(h_box, 1.0)
  cx, cy = w_box / 2.0, h_box / 2.0

  M_crop = np.eye(3)
  M_crop[0, 2] = -x1
  M_crop[1, 2] = -y1

  if angle is None:
    lnds_list = [[float(b_lnds[i][0]), float(b_lnds[i][1])] for i in range(len(b_lnds))]
    angle_deg = get_rot_angle(lnds_list[0], lnds_list[1])
  else:
    angle_deg = angle
  theta = np.radians(angle_deg)
  cos_a = np.cos(theta)
  sin_a = np.sin(theta)
  R = np.array([[cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]])

  T_to_center = np.eye(3); T_to_center[0, 2] = -cx; T_to_center[1, 2] = -cy
  T_from_center = np.eye(3); T_from_center[0, 2] = cx; T_from_center[1, 2] = cy
  M_rot_step = T_from_center @ R @ T_to_center

  M_scale = np.eye(3)
  M_scale[0, 0] = target_size[0] / w_box
  M_scale[1, 1] = target_size[1] / h_box

  M_total = M_scale @ M_rot_step @ M_crop
  return M_total[:2, :]


def get_torch_affine(M, src_shape, target_shape):
  """Convert a cv2-style pixel-space 2x3 affine into an F.affine_grid theta
  in the normalized [-1,1] coordinate frame. Includes the inverse-mapping
  step because grid_sample reads source pixels for each output location."""
  W, H = src_shape
  M_aug = np.vstack([M, [0, 0, 1]])
  M_inv = np.linalg.inv(M_aug)
  S_src = np.array([[2/W, 0, -1], [0, 2/H, -1], [0, 0, 1]])
  S_dst = np.array([[2/target_shape[0], 0, -1], [0, 2/target_shape[1], -1], [0, 0, 1]])
  M_torch = S_src @ M_inv @ np.linalg.inv(S_dst)
  return torch.from_numpy(M_torch[:2, :]).float()


def mi_on_affine(affine, batch_size, device, scale_jitter=0.1, shift_pixels=4.0, target=112.0):
  """Momentum-iterative input diversity: jitter the crop (scale +/- 10% and
  a few pixels of translation) rather than the perturbed pixels themselves."""
  scale_noise = 1.0 + (torch.rand(batch_size, 1, 1, device=device) - 0.5) * 2 * scale_jitter
  M_noisy = affine.clone()
  M_noisy[:, :, :2] = affine[:, :, :2] * scale_noise
  shift_noise = (torch.rand(batch_size, 2, 1, device=device) - 0.5) * (2 * shift_pixels / target * 2)
  M_noisy[:, :, 2:] = affine[:, :, 2:] + shift_noise
  return M_noisy


def get_grid(device, b_boxs, b_lnds=None, angle=None,
             src_shape=(112,112), target_shape=(112,112),
             batch_size=1, affine_arrays=None, mi_transform=True):
  if affine_arrays is None:
    theta_batch = torch.zeros((batch_size, 2, 3), device=device)
    for b in range(batch_size):
      M = get_affine(b_boxs[b], b_lnds=b_lnds[b], angle=angle, target_size=target_shape)
      theta = get_torch_affine(M, src_shape, target_shape).to(device)
      theta_batch[b] = theta.clone()
  else:
    theta_batch = affine_arrays
  noisy_theta = mi_on_affine(theta_batch, batch_size, device) if mi_transform else theta_batch
  grid = F.affine_grid(noisy_theta,
                       torch.Size((batch_size, 3, target_shape[1], target_shape[0])),
                       align_corners=True)
  return grid, theta_batch


def differentiable_warping(device, img, b_boxs=None, b_lnds=None, angle=None,
                           affine_arrays=None, mi_transform=True,
                           target_shape=(112,112)):
  batch_size = img.shape[0]
  grid, affines = get_grid(device, b_boxs, b_lnds=b_lnds, angle=angle,
                           src_shape=(img.shape[3], img.shape[2]),
                           target_shape=target_shape, batch_size=batch_size,
                           affine_arrays=affine_arrays, mi_transform=mi_transform)
  warped = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
  return warped, affines

