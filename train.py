import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

import config
from lib.dataset.codd import CODDAggDataset
from lib.dataset.kitti import KITTIOdometryDataset
from lib.dataset.datatransforms import Compose, VoxelSampling, RandomRotationTransform

def executeEpoch(model, loader, opt, sched, e, sw, mode='train'):
    assert mode == 'train' or mode =='val', 'mode should be train or val'

    if mode == 'train':
        model.train()
    else:
        model.eval()

    lE, lpE, lnE = 0, 0, 0
    rotE, transE = 0, 0
    maxInliersE, actualInliersE = 0, 0

    for b, (pb,RtGT) in enumerate(loader):
        pb = pb.cuda()
        RtGT = RtGT.cuda()

def train(args):
    #load dataset
    if args.dataset == 'codd':
        trainDataset = CODDAggDataset(config.CODD_PATH, mode='train', transform=Compose([VoxelSampling(0.3), RandomRotationTransform(rsig=40)]))
        valDataset = CODDAggDataset(config.CODD_PATH, mode='val', transform=VoxelSampling(0.3))
    elif args.dataset == 'kitti':
        trainDataset = KITTIOdometryDataset(config.KITTI_PATH, mode='train', transform=Compose([VoxelSampling(0.3), RandomRotationTransform(rsig=40)]))
        valDataset = KITTIOdometryDataset(config.KITTI_PATH, mode='val', transform=VoxelSampling(0.3))

    print(type(trainDataset))
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=config.batch_size, pin_memory=True, drop_last=True, num_workers=config.batch_size, shuffle=True)
    valLoader= torch.utils.data.DataLoader(valDataset, batch_size=config.batch_size, pin_memory=True, drop_last=True, num_workers=config.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains FastReg registration model')
    parser.add_argument('dataset', choices=('codd','kitti'), help='dataset used for evaluation')
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint (continue training)')
    args = parser.parse_args()
    train(args)