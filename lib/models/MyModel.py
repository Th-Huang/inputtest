import torch
import torch.nn.functional as F

from lib.pointnet2.pointnet2_utils import three_nn
from lib.pointnet2.pytorch_utils import SharedMLP

from lib.models.attention import GNNAttention
from lib.models.encoder import Encoder
from lib.utils.transforms import transformPoints, transformPointsBatch
from lib.models.decoder import Decoder

import open3d as o3d
import numpy as np

class _MyModelBase(torch.nn.Module):
    '''Base-class for registration network. Requires a point-wise encoder and a GNN to propagate point features. '''

    def __init__(self, T):
        super().__init__()

        self.debug = False
        self.T = T

    def encoder(self, pts):
        '''Encoder must accept pts [batch*2, npts, 4] and return [batch*2, npts, D] feature vectors.'''
        raise NotImplementedError()

    def graphNet(self, xyz0, xyz1, f0, f1):
        '''Computes graph attention. Must accept xyz[batch,npts,3] and f[batch,npts,D] and return new features f0a and f1a with same dimensions'''
        raise NotImplementedError()

    def decoder(self, xyz0):
        raise NotImplementedError()

    def forward(self, input, coord):

        batch_size = input.size(0)
        coord_xyz, coord_f = self.encoder(coord)


        # runs graphNet
        f0, f1 = self.graphNet(input, coord_xyz, coord_f)

        xyz = self.decoder(f0, f1)

        return xyz

class MyModel(_MyModelBase):
    def __init__(self, T):
        super().__init__(T)

        self.enc = Encoder()  # output feature dim: 128
        self.gnn = GNNAttention(dim=128, k=32)
        self.dec = Decoder(num_class=1000)

    def encoder(self, pts):
        return self.enc(pts)

    def graphNet(self, xyz0, xyz1, f0, f1):
        return self.gnn(xyz0, xyz1, f0, f1)

    def decoder(self, f0):
        return self.dec(f0)
