import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pywt
import pywt.data
from utils import convert_rgb_to_y
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor

# Foveated Wavelet Image Quality Index

g = [1.501, 1, 1, 0.534]
A = [[0.62171, 0.34537, 0.18004, 0.091401, 0.045943, 0.023013],
    [0.67234, 0.41317, 0.22727, 0.11792, 0.059758, 0.030018],
     [0.67234, 0.41317, 0.22727, 0.11792, 0.059758, 0.030018],
     [0.72709, 0.49428, 0.28688, 0.15214, 0.077727, 0.039156]]


# orientation: 0:LL, 1:LH, 2:HL, 3:HH
def _compute_sw(level, orientation, view_dist=1.0, screen_width=1920):
    alpha = 0.495
    k = 0.466
    f0 = 0.401
    A_sub = A[orientation][level]
    g_theta = g[orientation]
    N = screen_width
    v = view_dist
    r = np.pi * N * v / 180
    Y_sub = A_sub * alpha * np.power(10, k * (np.log10(2**level * f0 * g_theta / r))**2)
    sw = A_sub / Y_sub
    return sw


def _compute_sf(level, height, width, h_eye, w_eye, view_dist=1.0, screen_width=1920):
    # view_dist: viewing distance, measured in screen width, 1 by default
    # screen_width: width of screen, measured in pixels, 1920 by default
    N = screen_width
    v = view_dist
    CT = 1. / 64
    alpha = 0.106
    e2 = 2.3
    level += 1 # 0, 1, 2 -> 1, 2, 3

    h_mat = np.arange(height).reshape((height, 1)).repeat(width, axis=1)
    w_mat = np.arange(width).reshape((1, width)).repeat(height, axis=0)
    # re-project position from sub-band
    h_mat *= (2 ** level)
    w_mat *= (2 ** level)
    dx = np.sqrt((h_mat-h_eye)**2 + (w_mat-w_eye)**2)
    dx += 1e-6 # avoid dividing by zero

    e = np.arctan(dx / (N * v)) * 180 / np.pi # convert to degree
    fc = (e2 * np.log(1 / CT)) / ((e + e2) * alpha)

    r = np.pi * N * v / 180
    fd = r / 2
    gate = (fc < fd)
    fm = fc * gate + fd * (1 - gate)

    f = r * (2 ** -level)
    gate = (f <= fm)
    sf = gate * np.exp(- alpha * f * e / e2)
    return sf


def _compute_s(level, orientation, height, width, h_eye, w_eye, view_dist=1.0, screen_width=1920):
    beta1 = 1
    beta2 = 2.5
    sw = _compute_sw(level, orientation, view_dist, screen_width)
    sf = _compute_sf(level, height, width, h_eye, w_eye, view_dist, screen_width)
    s = (sw ** beta1) * (sf ** beta2)
    return s


def _compute_q(prediction, reference, device="cuda", window_size=11):
    window = torch.ones((1,1,window_size, window_size)).double().to(device)
    transform = Compose([ToTensor(), ])
    img1 = transform(prediction).double().to(device)
    img2 = transform(reference).double().to(device)
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    mu1 = F.conv2d(img1, window, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2

    q = (4 * sigma12 * mu1_mu2) / ((sigma1_sq + sigma2_sq) * (mu1_sq + mu2_sq))
    q = q.squeeze(0).squeeze(0)
    q = q.cpu().numpy()
    q[np.isnan(q)] = 0  # avoid nan, unstable when numerator is close to zero
    q = np.clip(q, -1, 1)
    return q


def _compute_fwqi_sub(pred_coeff, ref_coeff, level, orient, h_eye, w_eye, device, view_dist=1.0):
    height, width = pred_coeff.shape
    s = _compute_s(level, orient, height, width, h_eye, w_eye, view_dist=view_dist)
    q = _compute_q(pred_coeff, ref_coeff, device)
    c = np.abs(ref_coeff)
    deno = np.sum(s * c * q)
    nume = np.sum(s * c)
    return deno, nume


# crop the size of coefficient into power of 2
def subband_crop(coeffs, h, w, level=3):
    LL = coeffs[0]
    sh, sw = LL.shape
    th = int(h * 2 ** -3)
    tw = int(w * 2 ** -3)
    coeffs[0] =  LL[sh // 2 - th // 2: sh // 2 + th // 2, sw // 2 - tw // 2: sw // 2 + tw // 2]
    for i in range(1, level+1):
        LH, HL, HH = coeffs[i]
        sh, sw = LH.shape
        th = int(h * 2 ** - (4 - i))
        tw = int(w * 2 ** - (4 - i))
        LH = LH[sh // 2 - th // 2: sh // 2 + th // 2, sw // 2 - tw // 2: sw // 2 + tw // 2]
        HL = HL[sh // 2 - th // 2: sh // 2 + th // 2, sw // 2 - tw // 2: sw // 2 + tw // 2]
        HH = HH[sh // 2 - th // 2: sh // 2 + th // 2, sw // 2 - tw // 2: sw // 2 + tw // 2]
        coeffs[i] = (LH, HL, HH)
    return coeffs


def compute_fwqi(prediction, reference, h_eye, w_eye, device="cuda", subband=False, view_dist = 1.0, num_level=3):
    ref = convert_rgb_to_y(reference)
    pred = convert_rgb_to_y(prediction)
    origin_height, origin_width = ref.shape
    ref_coeff = pywt.wavedec2(ref, 'bior4.4', level=num_level)
    pred_coeff = pywt.wavedec2(pred, 'bior4.4', level=num_level)
    subband_crop(ref_coeff, origin_height, origin_width, num_level)
    subband_crop(pred_coeff, origin_height, origin_width, num_level)

    deno = 0
    nume = 0
    subband_fwqi = []

    pred_LL = pred_coeff[0]
    ref_LL = ref_coeff[0]
    deno_sub, nume_sub = _compute_fwqi_sub(pred_LL, ref_LL, num_level - 1, 0, h_eye, w_eye, device, view_dist)
    deno += deno_sub
    nume += nume_sub
    subband_fwqi.append(deno_sub / nume_sub)

    for l in range(1, num_level + 1):
        for o in range(3):
            pred_sub = pred_coeff[l][o]
            ref_sub = ref_coeff[l][o]
            deno_sub, nume_sub  = _compute_fwqi_sub(pred_sub, ref_sub, num_level - l, o+1, h_eye, w_eye, device, view_dist)
            deno += deno_sub
            nume += nume_sub
            subband_fwqi.append(deno_sub / nume_sub)

    total_fwqi = deno / nume
    if subband:
        return total_fwqi, subband_fwqi
    return total_fwqi


def compute_fwqi_tensor(prediction, reference, h_eye, w_eye, device="cuda", subband=False, view_dist = 1.0, num_level=3):
    total_fwqi = 0
    N, H, W, C = prediction.shape
    for i in range(N):
        pred = prediction[i].cpu().numpy()
        ref = reference[i].cpu().numpy()
        total_fwqi += compute_fwqi(pred, ref, h_eye[i], w_eye[i], device, subband, view_dist, num_level)
    return total_fwqi / N




