import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
import torch.nn.functional as F
from Gaussian_downsample import gaussian_downsample
from torchvision.transforms import Compose, ToTensor

def load_img(image_path, scale):
    HR = []
    for img_num in range(7):
        GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
        HR.append(GT_temp)
    return HR


def modcrop(img,scale):
    (ih, iw) = img.size
    ih = ih - ( ih % scale)
    iw = iw - ( iw % scale)
    img = img.crop((0,0,ih,iw))
    return img


# patch 64 * scale 4 = 256
# numpy data (T, H, W, C)
def random_crop(np_data, h_patch=256, w_patch=256):
    t, h, w, c = np_data.shape
    h_start = random.choice(list(range(0, h - h_patch + 1)))
    w_start = random.choice(list(range(0, w - w_patch + 1)))
    result = np_data[:, h_start:h_start + h_patch, w_start:w_start + w_patch, :]
    return result


def train_process(GH, flip_h=True, rot=True, converse=True):
    if random.random() < 0.5 and flip_h: # vertical flip
        GH = [ImageOps.flip(LR) for LR in GH]
    if random.random() < 0.5 and rot: # horizontal flip
        GH = [ImageOps.mirror(LR) for LR in GH]
    return GH


class TrainDataset(data.Dataset):
    def __init__(self, image_dir, scale, file_list):
        super(TrainDataset, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join(image_dir,file_list))]
        self.image_filenames = [os.path.join(image_dir,x) for x in alist]
        self.scale = scale
        self.transform = Compose([ToTensor(),])

    def __getitem__(self, index):
        GT = load_img(self.image_filenames[index], self.scale)
        GT = train_process(GT) # input: list (contain PIL), target: PIL
        GT = [np.asarray(HR) for HR in GT]  # PIL -> numpy # input: list (contatin numpy: [H,W,C])
        GT = np.asarray(GT) # numpy, [T,H,W,C]

        GT = random_crop(GT)

        # if self.scale == 4:
        #     GT = np.lib.pad(GT, pad_width=((0,0),(2*self.scale,2*self.scale),(2*self.scale,2*self.scale),(0,0)), mode='reflect')

        t, h, w, c = GT.shape
        GT = GT.transpose(1,2,3,0).reshape(h, w, -1) # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w) # Tensor, [C,T,H,W]
        LR = gaussian_downsample(GT, self.scale)
        LR = torch.cat((LR[:,1:2,:,:], LR), dim=1)
        return LR, GT

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    from torchvision.transforms import Compose, ToTensor
    image_dir = "D:/Datasets/vimeo_90k"
    file_list = "exp_trainlist.txt"
    dataloader = TrainDataset(image_dir, 4, file_list)
    print(len(dataloader))
    for i in range(len(dataloader)):
        lr, gt = dataloader[i]
        print(lr.shape)
        print(gt.shape)


