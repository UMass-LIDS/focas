from __future__ import print_function
import argparse
import os
import torch
from torch.utils.data import DataLoader
import sys
from utils import Logger, show_video_trace, display_config
import numpy as np
from tqdm import tqdm
from metric.ewpsnr import compute_ewpsnr, compute_ewpsnr_tensor
from metric.psnr import compute_psnr, compute_psnr_tensor
from metric.fwqi import compute_fwqi, compute_fwqi_tensor
from model.focas import focas_generator
from data.infer_dataset import InferDataset

def _infer_focas(model1, model2, lr, gaze, duration1, duration2, scale=4, n_c=128):
    out = []
    is_model1 = True
    duration_cnt = 0

    with torch.no_grad():
        init = True
        for i in range(len(gaze)):
            lr_input = lr[:, :, i:i + 2, :, :]
            h_eye, w_eye = gaze[i]
            if init:
                init_temp = torch.zeros_like(lr[:, 0:1, 0, :, :])
                init_o = init_temp.repeat(1, scale * scale * 3, 1, 1)
                init_h = init_temp.repeat(1, n_c, 1, 1)
                init_o = init_o.cuda()
                init_h = init_h.cuda()

                if is_model1:
                    h, old_pred = model1(lr_input, init_h, init_o, h_eye, w_eye, init)
                else:
                    h, old_pred = model2(lr_input, init_h, init_o, h_eye, w_eye, init)

                init = False
            else:
                if is_model1:
                    h, new_pred = model1(lr_input, h, old_pred, h_eye, w_eye, init)
                else:
                    h, new_pred = model2(lr_input, h, old_pred, h_eye, w_eye, init)
                out.append(old_pred)
                old_pred = new_pred

            duration_cnt += 1
            if is_model1:
                if duration_cnt == duration1:
                    is_model1 = False
                    duration_cnt = 0
            else:
                if duration_cnt == duration2:
                    is_model1 = True
                    duration_cnt = 0

        out.append(old_pred)

    torch.cuda.synchronize()

    out = torch.stack(out, dim=2)
    out = torch.clamp(out, 0, 1)
    return out


def infer(infer_loader, model1, model2, duration1, duration2, verbose=False, seg_size=50, fwqi=False):
    model1.eval()
    model2.eval()

    psnr_list = []
    ewpsnr_list = []
    fwqi_list = []

    data_enumerator = enumerate(tqdm(infer_loader)) if verbose else enumerate(infer_loader)
    for video_idx, data in data_enumerator:
        lr, hr, gaze = data

        video_psnr_list = []
        video_ewpsnr_list = []
        video_fwqi_list = []

        B, _, T, _ ,_ = lr.shape # B, C, T, H, W
        T = T - 1 # not include the padding frame

        hr = hr.squeeze(0)
        hr = hr.permute(1, 2, 3, 0)  # (T, H, W, C)

        start_idx = 0
        end_idx = seg_size

        while start_idx < T:
            if end_idx > T:
                end_idx = T

            lr_seg = lr[:, :, start_idx:end_idx+1, :, :].cuda()
            gaze_seg = gaze[start_idx:end_idx]
            pred = _infer_focas(model1, model2, lr_seg, gaze_seg, duration1, duration2)

            if pred is None:
                start_idx += seg_size
                end_idx += seg_size
                continue

            pred = pred.squeeze(0)
            pred = pred.permute(1, 2, 3, 0)  # (T, H, W, C)
            hr_seg = hr[start_idx:end_idx, :, :, :].cuda()

            pred *= 255.
            hr_seg *= 255.

            h_eye_list = []
            w_eye_list = []
            for i in range(start_idx, end_idx):
                h_eye, w_eye = gaze[i]
                h_eye = float(h_eye.numpy())
                w_eye = float(w_eye.numpy())
                h_eye_list.append(h_eye)
                w_eye_list.append(w_eye)

            video_psnr = compute_psnr_tensor(pred, hr_seg)
            video_ewpsnr = compute_ewpsnr_tensor(pred, hr_seg, h_eye_list, w_eye_list)
            if fwqi:
                video_fwqi = compute_fwqi_tensor(pred, hr_seg, h_eye_list, w_eye_list)

            video_psnr_list.append(video_psnr)
            video_ewpsnr_list.append(video_ewpsnr)
            if fwqi:
                video_fwqi_list.append(video_fwqi)

            start_idx += seg_size
            end_idx += seg_size

        video_psnr = float(np.mean(np.array(video_psnr_list)))
        psnr_list.append(video_psnr)
        video_ewpsnr = float(np.mean(np.array(video_ewpsnr_list)))
        ewpsnr_list.append(video_ewpsnr)
        if fwqi:
            video_fwqi = float(np.mean(np.array(video_fwqi_list)))
            fwqi_list.append(video_fwqi)


    print("Inference Results")

    print("PNSR")
    print(psnr_list)
    print("EWPSNR")
    print(ewpsnr_list)
    if fwqi:
        print("FWQI")
        print(fwqi_list)

    avg_psnr = np.mean(np.array(psnr_list))
    avg_ewpsnr = np.mean(np.array(ewpsnr_list))
    avg_fwqi = np.mean(np.array(fwqi_list))

    print("Average PSNR:", avg_psnr)
    print("Average EWPSNR:", avg_ewpsnr)
    if fwqi:
        print("Average FWQI:", avg_fwqi)

    return avg_psnr, avg_ewpsnr, avg_fwqi, psnr_list, ewpsnr_list, fwqi_list


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='FOCAS')
    parser.add_argument('--model1', default="RRN10", type=str, help="model1 name")
    parser.add_argument('--model2', default="FOCAS25", type=str, help="model2 name")
    parser.add_argument('--duration1', default=1, type=int, help="number of frames model1 is used for")
    parser.add_argument('--duration2', default=3, type=int, help="number of frames model2 is used for")
    parser.add_argument('--log_name', default="infer_test", type=str, help="log name")
    parser.add_argument('--data_dir', default="/mnt/nfs/work1/ramesh/lingdongwang/datasets/HD_UHD_EyeTracking/data/", type=str, help="dataset path")
    parser.add_argument('--model_path', default="pretrained/RRN-10L.pth", type=str, help="pretrained model path")
    parser.add_argument('--seg', default=50, type=int, help="video segment size")
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--fwqi', action='store_true', default=False, help="use FWQI metric")
    opt = parser.parse_args()

    sys.stdout = Logger(os.path.join('result', 'log', opt.log_name + '.txt'))
    display_config(opt)

    infer_dataset = InferDataset(opt.data_dir)
    infer_loader = DataLoader(dataset=infer_dataset, batch_size=1, shuffle=False,
                             pin_memory=False, drop_last=True)

    model1 = focas_generator(opt.model1, "HD", infer=True)
    model1.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.path).items()})
    model1.cuda()

    model2 = focas_generator(opt.model2, "HD", infer=True)
    model2.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.path).items()})
    model2.cuda()

    infer(infer_loader, model1, model2, duration1=opt.duration1, duration2=opt.duration2, verbose=opt.verbose, seg_size=opt.seg, fwqi=opt.fwqi)


