from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import functools
from model.rrn import RRN, make_layer, ResidualBlock_noBN, PixelUnShuffle, initialize_weights
from copy import deepcopy
import random

# linear interpolation mask to blend adjacent regions at the boundary
def blending_mask(N, C, H, W):
    mask = 0.2 * torch.ones((N, C, H, W))
    for i in range(1, 5):
        mask[:, :, i:-i, i:-i] = (i + 1) * 0.2
    return mask


# compute shapes of regions given the gaze point position
def compute_crop(regions, h_eye, w_eye):
    feat_crops = []
    mask_crops = []
    H, W = regions[0]

    for i in range(1, len(regions)):
        size = regions[i]
        left = w_eye - size[1] // 2
        right = w_eye + size[1] // 2
        up = h_eye - size[0] // 2
        down = h_eye + size[0] // 2

        left_short = 0
        right_short = 0
        up_short = 0
        down_short = 0

        if left < 0:
            left_short = 0 - left
            left = 0
        if right > W:
            right_short = W - right
            right = W
        if up < 0:
            up_short = 0 - up
            up = 0
        if down > H:
            down_short = H - down
            down = H

        mask_crops.append(([up_short, size[0] + down_short, left_short, size[1] + right_short]))
        feat_crops.append((up, down, left, right))

        h_eye -= up
        w_eye -= left
        H -= up
        W -= left

    return mask_crops, feat_crops


def crop(data, crop_pos):
    up, down, left, right = crop_pos
    return data[:, :, up:down, left:right]


# stack SR features from all regions together
def combine(feats, masks, feat_crops):
    inner = feats[-1]
    for i in range(len(feats)-2, -1, -1):
        outer = feats[i]
        mask = masks[i]
        up, down, left, right = feat_crops[i]
        outer_old = outer[:, :, up:down, left:right]
        outer[:, :, up:down, left:right] = inner * mask + outer_old * (1 - mask)
        inner = outer
    return inner


class neuro(nn.Module):
    def __init__(self, n_c, n_b, scale, blocks, regions, infer):
        super(neuro, self).__init__()
        pad = (1, 1)
        self.infer = infer
        self.scale = scale
        self.steps = [blocks[0]]
        for i in range(1,len(blocks)):
            self.steps.append(blocks[i] - blocks[i-1])
        self.blocks = blocks
        self.regions = [(r[0]//scale//2*2, r[1]//scale//2*2) for r in regions]

        self.conv_1 = nn.Conv2d(scale ** 2 * 3 + n_c + 3 * 2, n_c, (3, 3), stride=(1, 1), padding=pad)
        basic_block = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.recon_trunk = make_layer(basic_block, n_b)
        self.conv_h = nn.Conv2d(n_c, n_c, (3, 3), stride=(1, 1), padding=pad)
        self.conv_o = nn.Conv2d(n_c, scale ** 2 * 3, (3, 3), stride=(1, 1), padding=pad)
        initialize_weights([self.conv_1, self.conv_h, self.conv_o], 0.1)

        self.masks = []
        for i in range(1, len(regions)):
            mask = blending_mask(1, n_c, self.regions[i][0]//2*2, self.regions[i][1]//2*2)
            self.masks.append(mask.cuda())

    def set_infer(self, infer_flag):
        self.infer = infer_flag

    def forward(self, data, feat, out, h_eye=None, w_eye=None):
        if self.infer:
            return self.infer_forward(data, feat, out, h_eye, w_eye)
        else:
            return self.train_forward(data, feat, out)

    # foveated inference is deactivated when training
    def train_forward(self, data, feat, out):
        x = torch.cat((data, feat, out), dim=1)
        x = F.relu(self.conv_1(x))

        block_iter = self.recon_trunk.children()
        num_step = self.blocks[0]
        for i in range(num_step):
            block = next(block_iter)
            x = block(x)

        x_h = F.relu(self.conv_h(x))
        x_o = self.conv_o(x)
        return x_h, x_o

    def infer_forward(self, data, feat, out, h_eye, w_eye):
        h_eye = int(h_eye) // self.scale
        w_eye = int(w_eye) // self.scale

        mask_crops, feat_crops = compute_crop(self.regions, h_eye, w_eye)
        data = torch.cat((data, feat, out), dim=1)
        data = F.relu(self.conv_1(data))
        block_iter = self.recon_trunk.children()
        tmp_feats = []
        tmp_masks = []
        for i in range(len(self.steps)):
            if i != 0:
                data = crop(data, feat_crops[i-1])
                mask = crop(self.masks[i-1], mask_crops[i-1])
                tmp_masks.append(mask)
            step = self.steps[i]
            for j in range(step):
                block = next(block_iter)
                data = block(data)

            tmp_feats.append(data)

        data = combine(tmp_feats, tmp_masks, feat_crops)
        x_h = F.relu(self.conv_h(data))
        x_o = self.conv_o(data)
        return x_h, x_o


class FOCAS(nn.Module):
    def __init__(self, scale, n_c, n_b, blocks, regions, infer):
        super(FOCAS, self).__init__()
        self.n_c = n_c   # number of channels
        self.n_b = n_b   # number of residual blocks
        self.blocks = blocks  # feature depths (number of blocks used)
        self.regions = regions  # region sizes
        self.infer = infer  # inference phase or not
        self.neuro = neuro(n_c, n_b, scale, blocks, regions, infer)
        self.scale = scale  # upscale factor
        self.down = PixelUnShuffle(scale)


    def forward(self, x, x_h, x_o, h_eye, w_eye, init):
        _, _, T, _, _ = x.shape
        f1 = x[:, :, 0, :, :]
        f2 = x[:, :, 1, :, :]
        x_input = torch.cat((f1, f2), dim=1)
        if init:
            x_h, x_o = self.neuro(x_input, x_h, x_o, h_eye, w_eye)
        else:
            x_o = self.down(x_o)
            x_h, x_o = self.neuro(x_input, x_h, x_o, h_eye, w_eye)
        x_o = F.pixel_shuffle(x_o, self.scale) + F.interpolate(f2, scale_factor=self.scale, mode='bilinear',
                                                               align_corners=False)
        return x_h, x_o


def focas_generator(model_name, dataset, infer, scale=4, n_c=128, n_b=10):
    full_size = None
    if dataset == "V90K":
        full_size = (256, 256)
    elif dataset == "HD":
        full_size = (1080, 1920)
    else:
        print("Unknown Dataset", dataset)
        exit(-1)

    if "RRN" in model_name:
        depth = int(model_name[-2:])
        blocks = [depth]
        regions = [full_size]
        model = FOCAS(scale, n_c, n_b, blocks, regions, infer)
        return model
    elif model_name == "FOCAS15":
        blocks = [1, 4, 10]
        regions = [(1080, 1920), (224,224), (128,128)]
        model = FOCAS(scale, n_c, n_b, blocks, regions, infer)
        return model
    elif model_name == "FOCAS16":
        blocks = [1, 8, 10]
        regions = [(1080, 1920), (256,256), (224,224)]
        model = FOCAS(scale, n_c, n_b, blocks, regions, infer)
        return model
    elif model_name == "FOCAS17":
        blocks = [1, 4, 10]
        regions = [(1080, 1920), (368,368), (288,288)]
        model = FOCAS(scale, n_c, n_b, blocks, regions, infer)
        return model
    elif model_name == "FOCAS20":
        blocks = [1, 8, 10]
        regions = [(1080, 1920), (448,448), (416,416)]
        model = FOCAS(scale, n_c, n_b, blocks, regions, infer)
        return model
    elif model_name == "FOCAS25":
        blocks = [1, 3, 10]
        regions = [(1080, 1920), (824,824), (544,544)]
        model = FOCAS(scale, n_c, n_b, blocks, regions, infer)
        return model
    else:
        print("Unknown Model", model_name)
        exit(-1)



if __name__ == '__main__':
    infer = False
    h_eye = 58*4
    w_eye = 434*4

    scale = 4
    n_c = 128
    n_b = 10
    N = 1
    H = 1080
    W = 1920
    H_lr = H // scale
    W_lr = W // scale
    model_path = "../pretrained/RRN-10L.pth"

    model = focas_generator("FOCAS25", "HD", infer=infer)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
    model.cuda()

    data = torch.ones((1, 3, 2, H_lr, W_lr))
    for i in range(1):
        init_temp = torch.zeros_like(torch.ones((N, 1, H_lr, W_lr)))  # (N, 1, H, W)
        init_temp = init_temp.cuda()
        out = init_temp.repeat(1, scale * scale * 3, 1, 1)  # (N, C=3scale^2, H, W)
        feat = init_temp.repeat(1, n_c, 1, 1)  # (N, C=n_c, H, W)
        out = out.cuda()
        feat = feat.cuda()
        data = data.cuda()
        print("data", data.shape, "feat", feat.shape, "out", out.shape)
        if infer:
            h, prediction = model(data, feat, out, h_eye, w_eye, init=True)
        else:
            h, prediction = model(data, feat, out, None, None, init=True)
        print("h", h.shape)
        print("out", prediction.shape)
