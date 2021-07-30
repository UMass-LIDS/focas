import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import random
from torchvision.transforms import Compose, ToTensor
import torch.nn.functional as F
from Gaussian_downsample import gaussian_downsample
import cv2
import csv
from tqdm import tqdm

# HD Video Observers
hd_obs_list = [5,6,7,8,9,12,13,14,15,16]
hd_obs_list.extend(list(range(19,43)))


def read_bgr_video(source_path, height=1080, width=1920, num_frame=None):
    file_size = os.path.getsize(source_path)
    frame_size = width * height * 3
    if num_frame is None:
        num_frame = file_size // frame_size
    # print("Number of Frames:", num_frame)
    f = open(source_path, "rb")
    data = f.read(frame_size * num_frame)
    f.close()

    data = np.frombuffer(data, dtype=np.uint8)
    data = np.array(data).reshape((num_frame, height, width, 3)).astype(np.uint8)
    b = data[:, :, :, 0]
    g = data[:, :, :, 1]
    r = data[:, :, :, 2]
    data = np.stack([r, g, b], axis=3)
    return data


def read_gaze_data(data_dir, video_name, obs_idx):
    file_name = "observer"+str(hd_obs_list[obs_idx])+"_"+video_name[:-4]+".csv" # observer 5 - 42
    file_path = os.path.join(data_dir, file_name)
    gaze_data = []
    reader = csv.reader(open(file_path))
    while True:
        try:
            left = next(reader)
            right = next(reader)
            if "NaN" in left or "NaN" in right:
                continue
            frame_idx = int(float(left[1])) // 40  # 25 fps video -> 40 ms interval
            h_eye = int(float(left[3]) + float(right[3])) // 2
            w_eye = int(float(left[2]) + float(right[2])) // 2
            gaze_data.append((frame_idx, h_eye, w_eye))
        except IndexError:
            continue
        except StopIteration:
            break
    return gaze_data


def valid_pos(h_eye, w_eye, H=1080, W=1920):
    if h_eye < 0:
        h_eye = 0
    if h_eye > H:
        h_eye = H
    if w_eye < 0:
        w_eye = 0
    if w_eye > W:
        w_eye = W
    return h_eye, w_eye


class InferDataset(data.Dataset):
    def __init__(self, data_dir, scale=4):
        super(InferDataset, self).__init__()
        print("Loading Inference Dataset ...")
        self.scale = scale
        self.video_dir = os.path.join(data_dir, "Videos", "HD")
        self.gaze_dir = os.path.join(data_dir, "Gaze_Data", "HD")
        self.video_names = os.listdir(self.video_dir)
        self.video_names.sort()
        self.num_videos = len(self.video_names)
        print("Number of Videos", self.num_videos)
        self.hr = None
        self.lr = None
        self.video_index = -1

    def __getitem__(self, idx):
        num_obs = len(hd_obs_list)
        video_idx = idx // num_obs
        obs_idx = idx % num_obs
        video_name = self.video_names[video_idx]

        gaze_data = read_gaze_data(self.gaze_dir, video_name, obs_idx)

        if video_idx != self.video_index:
            self.video_index = video_idx
            hr_data = read_bgr_video(os.path.join(self.video_dir, video_name,))
            T, H, W, C = hr_data.shape
            hr_data = hr_data.transpose(1, 2, 3, 0).reshape(H, W, -1)  # numpy, [H',W',CT]
            transform = Compose([ToTensor(), ])
            hr_data = transform(hr_data)
            hr_data = hr_data.view(C, T, H, W)
            lr_data = gaussian_downsample(hr_data, self.scale)

            lr_data = torch.cat([lr_data[:,0:1,:,:], lr_data], dim=1)  # duplicate the first frame for SR input
            self.lr = lr_data
            self.hr = hr_data

        C, T, H, W = self.hr.shape
        gaze = [None for _ in range(T)]
        for data_idx in range(len(gaze_data)-1, -1, -1):
            frame_idx, h_eye, w_eye = gaze_data[data_idx]
            if frame_idx >= T:
                frame_idx = T - 1
            h_eye, w_eye = valid_pos(h_eye, w_eye)
            gaze[frame_idx] = (h_eye, w_eye)
        for i in range(T):
            if gaze[i] is None:
                gaze[i] = (540, 960)  # assume gaze point to be center when losing track

        return self.lr, self.hr, gaze

    def __len__(self):
        return self.num_videos * len(hd_obs_list)


if __name__ == '__main__':
    from utils import show_video_trace, show_video
    import time

    data_dir = "D:/Datasets/HD_UHD_Eyetracking/"

    # gaze_dir = os.path.join(data_dir, "Gaze_Data", "HD")
    # gaze_data = read_gaze_data(gaze_dir, 'Wood_s1920x1080p25n300v0.bgr', 5)
    # print(gaze_data)
    # print(len(gaze_data))

    # video_data = read_bgr_video(data_dir+video_name)
    # print(video_data.shape)

    dataloader = InferDataset(data_dir)
    print("Data Loader Size", len(dataloader))
    for i in range(len(dataloader)):
        st = time.time()
        lr, hr, gaze = dataloader[i]
        print(lr.shape)
        print(hr.shape)
        print(len(gaze))
        print(gaze)

        # print("time", time.time()-st)

        # show_video_trace(hr, gaze, title="wood_hr.mp4")
        # show_video(hr, title="wood_hr.mp4")
        # lr = lr[:, 1:, :, :]
        # gaze = [(x[0]//4, x[1]//4) for x in gaze]
        # show_video(lr, size=(480, 270), title="wood_lr.mp4")
        # show_video_trace(lr, gaze, size=(480,270), point_size=2, title="wood_lr.mp4")
        if i == 5:
            break




