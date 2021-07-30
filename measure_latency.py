from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import functools
from time import time
from model.focas import FOCAS, focas_generator


# measure the latency of FOCAS model
def latency_focas(blocks, regions, n_b=10, H = 1080, W=1920, n_c=128, scale=4, num_frame=20, num_iter=20, warm_up=5):
    HH = H // scale
    WW = W // scale

    model = FOCAS(scale, n_c, n_b, blocks=blocks, regions=regions, infer=True)
    model.cuda()
    start_time = None
    out_list = []
    feat_list = []
    data_list = []
    for i in range(num_iter):
        if i == warm_up:
            start_time = time()
        init_temp = torch.zeros_like(torch.ones((1, 1, HH, WW)))  # (N, 1, H, W)
        init_temp = init_temp.cuda()
        out = init_temp.repeat(1, scale * scale * 3, 1, 1)  # (N, C=3scale^2, H, W)
        feat = init_temp.repeat(1, n_c, 1, 1)  # (N, C=n_c, H, W)
        data = torch.randn((1, 3, 2, HH, WW))
        out = out.cuda()
        feat = feat.cuda()
        data = data.cuda()
        out_list.append(out)
        feat_list.append(feat)
        data_list.append(data)

    for i in range(num_iter):
        # warm-up iterations before warm_up will be ignored
        if i == warm_up:
            start_time = time()
        with torch.no_grad():
            out = out_list[i]
            feat = feat_list[i]
            data = data_list[i]
            init_flag = True

            for j in range(num_frame):
                feat, out = model(data, feat, out, H//2, W//2, init_flag)
                if init_flag:
                    init_flag = False

    run_time = time() - start_time
    run_time /= num_iter-warm_up
    run_time /= num_frame
    return run_time


# measure latencies of sampled FOCAS models for latency estimation
def latency_focas_fit(verbose=False):
    print("Measure Model Latency for Fitting ...")
    result = []
    for b1 in [1,2,3]:
        for b2 in [4,5,6]:
            for b3 in [7,8,9]:
                for s1 in [80*4, 100*4]:
                    for s2 in [40*4, 60*4]:
                        blocks = (b1, b2, b3)
                        regions = ((1080, 1920), (s1, s1), (s2, s2))
                        latency = latency_focas(blocks=blocks, regions=regions)
                        result.append((b1, b2, b3, s1, s2, latency))
                        if verbose:
                            print("Block1, Block2, Block3, Size1, Size2, Latency")
                            print(b1, b2, b3, s1, s2, latency)
    return result


# measure latencies of FOCAS models proved in the paper
def latency_focas_paper():
    for model_name in ["FOCAS25", "FOCAS20", "FOCAS17", "FOCAS16", "FOCAS15"]:
        model = focas_generator(model_name, "HD", infer=True)
        latency = latency_focas(blocks=model.blocks, regions=model.regions)
        print("model name", model_name, "latency", latency)


if __name__ == '__main__':
    img_height = 1080
    img_width = 1920

    result = latency_focas_fit()
    print(result)