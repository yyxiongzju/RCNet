import argparse
import csv
import os
import random
import sys
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange

import flops_benchmark
from data import get_loaders
from models import *
from run import test

parser = argparse.ArgumentParser(description='RCNet training with PyTorch')
parser.add_argument('--dataroot', required=True, metavar='PATH',
                    help='Path to ImageNet train and val folders, preprocessed as described in '
                         'https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset')
parser.add_argument('--gpus', default=None, help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')
parser.add_argument('--classes', type=int, default=1000, help= "Number of catogories")

parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--load', default='../../trained_models/rcnet_imagenet.pth.tar', type=str, metavar='PATH', help='path to trained model (default: none)')
parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of RCNet (default x1).')
parser.add_argument('--input-size', type=int, default=224, metavar='I',
                    help='Input size of RCNet, multiple of 32 (default 224).')

def create_arch4net(bin_setting, arch_setting):
    cfg = [(1,  16, 1, 1, 1, 1)]
    for i in range(len(bin_setting)):
        if bin_setting[i] == 1:
            cfg.append((3, arch_setting[i][0], arch_setting[i][1], arch_setting[i][2], 1, 1))
        elif bin_setting[i] == 2:
            cfg.append((3, arch_setting[i][0], arch_setting[i][1], arch_setting[i][2], 1, 2))
        elif bin_setting[i] == 3:
            cfg.append((6, arch_setting[i][0], arch_setting[i][1], arch_setting[i][2], 2, 2))
        elif bin_setting[i] == 4:
            cfg.append((6, arch_setting[i][0], arch_setting[i][1], arch_setting[i][2], 1, 2))
        elif bin_setting[i] == 5:
            cfg.append((6, arch_setting[i][0], arch_setting[i][1], arch_setting[i][2], 1, 1))
        elif bin_setting[i] == 6:
            cfg.append((12, arch_setting[i][0], arch_setting[i][1], arch_setting[i][2], 2, 4))
        elif bin_setting[i] == 0:
            continue
        else:
            print("Warnning: no block type matched!")
    return cfg

def main():
    args = parser.parse_args()

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'

    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8


    arch_setting = [(24, 1, 2),
                    (24, 1, 1),
                    (24, 1, 1),
                    (24, 1, 1),
                    (24, 1, 1),
                    (24, 1, 1),
                    (32, 1, 2),
                    (32, 1, 1),
                    (32, 1, 1),
                    (32, 1, 1),
                    (32, 1, 1),
                    (32, 1, 1),
                    (64, 1, 2),
                    (64, 1, 1),
                    (64, 1, 1),
                    (64, 1, 1),
                    (64, 1, 1),
                    (64, 1, 1),
                    (96, 1, 1),
                    (96, 1, 1),
                    (96, 1, 1),
                    (96, 1, 1),
                    (96, 1, 1),
                    (96, 1, 1),
                    (160, 1, 2),
                    (160, 1, 1),
                    (160, 1, 1),
                    (160, 1, 1),
                    (160, 1, 1),
                    (160, 1, 1),
                    (320, 1, 1),
                    (320, 1, 1),
                    (320, 1, 1),
                    (320, 1, 1),
                    (320, 1, 1),
                    (320, 1, 1)]

    print ("Evaluate RCNet on ImageNet")
    bin_setting = [5, 3, 5, 1, 0, 0, 5, 2, 5, 3, 3, 5, 5, 5, 3, 5, 5, 6, 5, 1, 5, 5, 4, 5, 5, 4, 5, 5, 4, 5, 0, 5, 0, 0, 0, 0]
    cfg4net = create_arch4net(bin_setting, arch_setting)
    model = RCNet(cfg4net)
    file_handler = RCNet
    
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print(model)
    print('number of parameters: {}'.format(num_parameters))
    print('FLOPs: {}'.format(
        flops_benchmark.count_flops(file_handler,
                                    args.batch_size // len(args.gpus) if args.gpus is not None else args.batch_size,
                                    device, dtype, args.input_size, 3, args.scaling, cfg4net)))


    train_loader, test_loader = get_loaders(args.dataroot, args.batch_size, args.batch_size, args.input_size,
                                           args.workers)
    # define loss function (criterion)
    criterion = torch.nn.CrossEntropyLoss()
    if args.gpus is not None:
        model = torch.nn.DataParallel(model, args.gpus)
    model.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        loss, top1, top5 = test(model, test_loader, criterion, device, dtype)
        print("=> Test loss: {},  Top 1 accu: {}, Top 5 accu: {}").format(loss, top1, top5)
        return

if __name__ == '__main__':
    main()
