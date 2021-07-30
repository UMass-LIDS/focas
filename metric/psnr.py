import numpy as np
import math
from PIL import Image
import torch
from utils import convert_rgb_to_y


def compute_psnr(prediction, target):
    # prediction and target have range [0, 255]
    img1 = prediction.astype(np.float64)
    img1 = convert_rgb_to_y(img1)
    img2 = target.astype(np.float64)
    img2 = convert_rgb_to_y(img2)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0**2 / mse)


def compute_psnr_tensor(pred, ref):
    pred = pred.float()
    ref = ref.float()
    pred = convert_rgb_to_y(pred)
    ref = convert_rgb_to_y(ref)
    mse = torch.mean((pred - ref) ** 2, dim=(1, 2))
    psnr = 10 * torch.log10(255.0 ** 2 / mse)
    return float(torch.mean(psnr))


