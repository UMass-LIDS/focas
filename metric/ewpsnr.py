import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from utils import convert_rgb_to_y
import torch

# visual attention guided bit allocation in video compression
# eye-tracking-weighted PSNR

# sigma is set to be 2 degree (64 pixels by default)
def compute_ewmse(prediction, reference, h_eye, w_eye, sigma_h=64, sigma_w=64):
    img1 = prediction.astype(np.float64)
    img1 = convert_rgb_to_y(img1)
    img2 = reference.astype(np.float64)
    img2 = convert_rgb_to_y(img2)
    H, W = img1.shape
    h_mat = np.arange(H).reshape((H,1)).repeat(W, axis=1)
    w_mat = np.arange(W).reshape((1,W)).repeat(H, axis=0)

    weight = np.exp(-((h_mat - h_eye)**2)/(2*sigma_h**2) - ((w_mat - w_eye)**2)/(2*sigma_w**2))
    weight /= 2 * np.pi * sigma_h * sigma_w

    ewmse = np.sum(weight * (img1-img2)**2)
    ewmse /= H*W*np.sum(weight)
    return ewmse


def compute_ewpsnr(prediction, reference, h_eye, w_eye, sigma_h=64, sigma_w=64):
    ewmse = compute_ewmse(prediction, reference, h_eye, w_eye, sigma_h, sigma_w)
    if ewmse == 0:
        return float('inf')
    return 10 * np.log10(255.0**2 / ewmse)


def compute_ewpsnr_tensor(pred, ref, h_eye, w_eye, sigma_h=64., sigma_w=64.):
    pred = pred.float()
    ref = ref.float()
    img1 = convert_rgb_to_y(pred)
    img2 = convert_rgb_to_y(ref)
    N, H, W = img1.shape
    h_eye = torch.FloatTensor(h_eye).reshape((N, 1, 1))
    w_eye = torch.FloatTensor(w_eye).reshape((N, 1, 1))

    h_mat = torch.from_numpy(np.arange(H).reshape((H, 1)).repeat(W, axis=1)).unsqueeze(0).repeat((N,1,1))
    w_mat = torch.from_numpy(np.arange(W).reshape((1, W)).repeat(H, axis=0)).unsqueeze(0).repeat((N,1,1))

    weight = torch.exp(-((h_mat - h_eye) ** 2) / (2 * sigma_h ** 2) - ((w_mat - w_eye) ** 2) / (2 * sigma_w ** 2))
    weight /= 2 * math.pi * sigma_h * sigma_w
    weight = weight.to(pred.device)

    ewmse = torch.sum(weight * (img1 - img2) ** 2, dim=(1,2))
    ewmse /= H * W * torch.sum(weight, dim=(1,2))
    return float(torch.mean(10 * torch.log10(255.0**2 / ewmse)))
