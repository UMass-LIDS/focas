from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import time
import sys
from utils import Logger, display_config
import numpy as np
from model.rrn import RRN
from model.focas import focas_generator
from training.valid import valid
from training.train import train
from data.train_dataset import TrainDataset

parser = argparse.ArgumentParser(description='FOCAS')
# basic setting
parser.add_argument('--model', type=str ,default='RRN10', help="model name")
parser.add_argument('--log_name', type=str, default='rrn10', help="log name")
parser.add_argument('--dataset', type=str ,default='V90K', help="dataset name")
parser.add_argument('--data_dir', type=str ,default='/mnt/nfs/work1/ramesh/lingdongwang/datasets/vimeo_septuplet/sequences/', help="dataset path")
parser.add_argument('--no_rec', action='store_true', default=False, help="not using recurrent state")
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=70, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots. This is a savepoint, using to save training model.')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--save_model_path', type=str, default='./result/weight', help='location to save checkpoint models')
parser.add_argument('--pretrained_path', type=str, default='pretrained/RRN-10L.pth', help="path of the pretrained model for evaluating")
parser.add_argument('--verbose', action='store_true', default=False)

# training
parser.add_argument('--train_list', type=str, default='sep_trainlist.txt', help="list of examples used for training")
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--stepsize', type=int, default=60, help='Learning rate is decayed by a factor of 10 every half of total epochs')
parser.add_argument('--gamma', type=float, default=0.1 , help='learning rate decay')
parser.add_argument('--log_path', type=str ,default='./result/log/', help="log path")
parser.add_argument('--weight-decay', default=5e-04, type=float,help="weight decay (default: 5e-04)")
parser.add_argument('--data_augmentation', type=bool, default=True, help="using data augmentation")
parser.add_argument('--retrain', action='store_true', default=False, help="retraining based on a pretrained model")

# validation
parser.add_argument('--valid_list', type=str, default='sep_testlist.txt', help="list of examples used for validation")
parser.add_argument('--valid_every', type=int, default=10, help="frequency of validation")

# testing
parser.add_argument('--test', action='store_true', default=False, help="whether is testing or not")
parser.add_argument('--test_list', type=str, default='sep_testlist.txt', help="list of examples used for testing")


opt = parser.parse_args()


def main():
    sys.stdout = Logger(os.path.join(opt.log_path, opt.log_name + '.txt'))
    torch.manual_seed(opt.seed)
    gpu_devices = os.environ['CUDA_VISIBLE_DEVICES']
    gpu_devices = gpu_devices.split(',')
    print("Using GPU", gpu_devices)

    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')


    cudnn.benchmark = True
    torch.cuda.manual_seed_all(opt.seed)
    pin_memory = True
    display_config(opt)

    # model setting
    model = focas_generator(opt.model, opt.dataset, infer=False)

    # load pre-trained model
    if opt.test or opt.retrain:
        try:
            model.load_state_dict(torch.load(opt.pretrained_path))
        except RuntimeError:  # in case of version mismatch
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.pretrained_path).items()})
        print("Pretrained Model Loaded")

    p = sum(p.numel() for p in model.parameters()) * 4 / 1048576.0  # 32 bit -> 4 Byte, 1024*1024 B = 1 MB
    print('Model Size: {:.2f}M'.format(p))

    if len(gpu_devices) > 1:
        model = torch.nn.DataParallel(model)
    criterion = nn.L1Loss(reduction='sum')

    model = model.cuda()
    criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    # dataset setting
    print('Loading Dataset ...')
    if opt.test:
        test_set = TrainDataset(opt.data_dir, opt.scale, opt.test_list)
        test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False,
                                 pin_memory=pin_memory, drop_last=True)

        valid(test_loader, model, verbose=opt.verbose, no_rec=opt.no_rec)

    else:  # training / re-training
        train_set = TrainDataset(opt.data_dir, opt.scale, opt.train_list)
        train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchsize, shuffle=True, pin_memory=pin_memory, drop_last=True)
        valid_set = TrainDataset(opt.data_dir, opt.scale, opt.valid_list)
        valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=1, shuffle=False, pin_memory=pin_memory, drop_last=True)

        for epoch in range(1, opt.nEpochs+1):
            train(train_loader, model, criterion, optimizer, epoch, verbose=opt.verbose)
            if (epoch+1) % opt.valid_every == 0:
                valid(valid_loader, model, verbose=opt.verbose)
            if opt.stepsize > 0:
                scheduler.step()
            if (epoch+1) % (opt.snapshots) == 0:
                checkpoint(model, epoch)
            sys.stdout.flush()


def checkpoint(model, epoch):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name)
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name = 'X'+str(opt.scale)+'_{}'.format(opt.model)+'_epoch_{}.pth'.format(epoch)
    torch.save(model.rrn_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))


if __name__ == '__main__':
    main()    
