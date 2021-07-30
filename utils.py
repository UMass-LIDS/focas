import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import pynvml
import os
import sys
import errno
import os.path as osp
import torch
import warnings
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_video(data, title="video.mp4", size=(1920, 1080)):
    print("Generating Eye-Tracing Video ... ")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # mp4
    video = cv2.VideoWriter(title, fourcc, 25, size)

    data = data.numpy()
    C, T, H, W = data.shape
    data *= 255.
    for t in tqdm(range(T)):
        # ignore exceeding boundary
        try:
            frame = data[:, t, :, :]
            frame = frame.transpose(1, 2, 0)
            frame = np.uint8(frame)
            # OpenCV uses BGR format
            frame = frame[:, :, ::-1]
            video.write(frame)
        except:
            pass
    video.release()
    print("Eye-Tracing Video Generated")


def show_video_trace(data, gaze, title="video.mp4", size=(1920, 1080), point_size=8):
    print("Generating Eye-Tracing Video ... ")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # mp4
    video = cv2.VideoWriter(title, fourcc, 25, size)

    data = data.numpy()
    C, T, H, W = data.shape
    data *= 255.
    red_point = np.zeros((point_size,point_size,3))
    half = point_size//2
    red_point[:, :, 0] = 255.
    for t in tqdm(range(T)):
        # ignore exceeding boundary
        try:
            frame = data[:, t, :, :]
            h_eye, w_eye = gaze[t]
            frame = frame.transpose(1, 2, 0)
            frame[h_eye-half:h_eye+half, w_eye-half:w_eye+half, :] = red_point
            frame = np.uint8(frame)
            # OpenCV uses BGR format
            frame = frame[:, :, ::-1]
            video.write(frame)
        except:
            pass
    video.release()
    print("Eye-Tracing Video Generated")


def circle_mask(diam):
    data = np.arange(diam).reshape((1, diam)) - diam / 2
    data = data.repeat(diam, axis=0)
    x2 = data * data
    y2 = data.T * data.T
    mask = ((x2 + y2) <= diam * diam / 4)
    return np.uint8(mask)


def video2img(video_name, video_path, image_path, num_frame=150):
    cap = cv2.VideoCapture(video_path + video_name)
    fps =cap.get(cv2.CAP_PROP_FPS)
    print("fps:",fps)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("size:",size)
    i=0
    video_name = video_name[:-4] # remove suffix
    if not os.path.exists(image_path + video_name):
        os.mkdir(image_path + video_name)
    while(cap.isOpened() and i < num_frame):
        i=i+1
        ret, frame = cap.read()
        if ret==True:
            cv2.imwrite(image_path+video_name+'/'+ 'hr_' + str(i).zfill(3)+'.jpg',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def img2video(path, width=3840, height=1920, fps=30, key_word=''):
    print("Converting Images to Video ...")
    img_paths = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        if key_word in img_path and file[-3:] != 'mp4':
            img_paths.append(img_path)
    print("Number of Frames:", len(img_paths))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # mp4
    size = (width, height)
    video = cv2.VideoWriter(os.path.join(path, "video.mp4"), fourcc, fps, size)

    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        video.write(img)
    video.release()


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def model_size(model):
    p = sum(p.numel() for p in model.parameters()) * 4 / 1048576.0
    return p


def display_config(args):
    print('-------SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')


def gpu_info():
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        print("GPU", i, ":", pynvml.nvmlDeviceGetName(handle))
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("Memory Total: {:.2f} G".format(info.total/1024**3))
        print("Memory Free: {:.2f} G".format(info.free/1024**3))
        print("Memory Used: {:.2f} G".format(info.used/1024**3))
    pynvml.nvmlShutdown()


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()