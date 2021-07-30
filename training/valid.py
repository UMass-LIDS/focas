from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import torch.backends.cudnn as cudnn
import cv2
import math
import sys
import datetime
from utils import Logger
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm


def valid(valid_loader, model, scale=4, n_c=128, verbose=False, no_rec=False):
    h_eye = None
    w_eye = None

    model.eval()
    count = 0
    PSNR_t = 0
    SSIM_t = 0

    if log != '':
        file = open(log+'.txt', 'w')

    data_enumerator = enumerate(tqdm(valid_loader)) if verbose else enumerate(valid_loader)
    for image_num, data in data_enumerator:
        x_input, target = data[0], data[1]
        out = []
        PSNR = 0
        SSIM = 0

        B, _, T, _ ,_ = x_input.shape
        T = T - 1  # not include the padding frame
        with torch.no_grad():
            x_input = Variable(x_input).cuda()
            target = Variable(target).cuda()
            # t0 = time.time()
            init = True
            for i in range(T):
                if init:
                    init_temp = torch.zeros_like(x_input[:,0:1,0,:,:])
                    init_o = init_temp.repeat(1, scale*scale*3, 1, 1)
                    init_h = init_temp.repeat(1, n_c, 1, 1)
                    h, prediction = model(x_input[:, :, i:i + 2, :, :], init_h, init_o, h_eye, w_eye, init)
                    out.append(prediction)
                    # not using recurrent state: always initializing it
                    if not no_rec:
                        init = False
                else:
                    h, prediction = model(x_input[:, :, i:i + 2, :, :], h, prediction, h_eye, w_eye, init)
                    out.append(prediction)
        torch.cuda.synchronize()
        # t1 = time.time()
        # print("===> Timer: %.4f sec." % (t1 - t0))
        prediction = torch.stack(out, dim=2)
        count += 1
        prediction = prediction.squeeze(0).permute(1,2,3,0)  # [T,H,W,C]
        prediction = prediction.detach().cpu().numpy()[:,:,:,::-1]  # tensor -> numpy, rgb -> bgr
        target = target.squeeze(0).permute(1,2,3,0)  # [T,H,W,C]
        target = target.cpu().numpy()[:,:,:,::-1]  # tensor -> numpy, rgb -> bgr
        target = crop_border_RGB(target, 8)
        prediction = crop_border_RGB(prediction, 8)
        for i in range(T):
            prediction_Y = bgr2ycbcr(prediction[i])
            target_Y = bgr2ycbcr(target[i])
            prediction_Y = prediction_Y * 255
            target_Y = target_Y * 255

            PSNR += calculate_psnr(prediction_Y, target_Y)
            SSIM += calculate_ssim(prediction_Y, target_Y)
            out.append(calculate_psnr(prediction_Y, target_Y))
        PSNR_t += PSNR / T
        SSIM_t += SSIM / T

    PSNR_t /= count
    SSIM_t /= count
    print('Valid PSNR = {}'.format(PSNR_t))
    print('Valid SSIM = {}'.format(SSIM_t))
    return PSNR_t, SSIM_t

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    Output:
        type is same as input
        unit8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def crop_border_Y(prediction, shave_border=0):
    prediction = prediction[shave_border:-shave_border, shave_border:-shave_border]
    return prediction


def crop_border_RGB(target, shave_border=0):
    target = target[:,shave_border:-shave_border, shave_border:-shave_border,:]
    return target


def calculate_psnr(prediction, target):
    # prediction and target have range [0, 255]
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
