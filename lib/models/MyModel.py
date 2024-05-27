import torch
import torch.nn.functional as F

from lib.pointnet2.pointnet2_utils import three_nn
from lib.pointnet2.pytorch_utils import SharedMLP

from lib.models.attention import GNNAttention
from lib.models.encoder import Encoder
from lib.utils.transforms import transformPoints, transformPointsBatch
from lib.models.decoder import Decoder
from lib.pointnet2.pointnet_modules import PointNetfeat
from lib.models.graphmatchingnetwork import GraphEmbeddingNet
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

    def point_decoder_net(self, pts):
        raise NotImplementedError()

    def graphNet(self, xyz0, xyz1, f0, f1):
        '''Computes graph attention. Must accept xyz[batch,npts,3] and f[batch,npts,D] and return new features f0a and f1a with same dimensions'''
        raise NotImplementedError()

    def decoder(self, xyz, xyz1, f1, xyz2, f2, xyz3, f3, xyz4, f4):
        raise NotImplementedError()

    def forward(self, input, coord):

        batch_size = input.size(0)
        # runs encoder
        xyz, xyz1, f1, xyz2, f2, xyz3, f3, xyz4, f4, f0 = self.encoder(coord)
        coord_c = xyz3
        coord_f = f0
        input_f = self.point_decoder_net(input.transpose(2, 1))
        # runs graphNet
        input_f, coord_f = self.graphNet(input, coord_c, input_f, coord_f)
        input_f,coord_f = self.graphmatching()
        # runs decoder
        coord_out = self.decoder(xyz, xyz1, f1, xyz2, f2, xyz3, coord_f.transpose(2,1), xyz4, f4)

        return coord_out

class MyModel(_MyModelBase):
    def __init__(self, T):
        super().__init__(T)

        self.enc = Encoder()  # output feature dim: 128
        self.point_decoder = PointNetfeat(global_feat=True, feature_transform=False)
        self.gnn = GNNAttention(dim=128, k_coord=32, k_input=5)
        self.dec = Decoder(num_class=3)

    def encoder(self, pts):
        return self.enc(pts)

    def point_decoder_net(self, pts):
        return self.point_decoder(pts)

    def graphNet(self, xyz0, xyz1, f0, f1):
        return self.gnn(xyz0, xyz1, f0, f1)

    def decoder(self, xyz, xyz1, f1, xyz2, f2, xyz3, f3, xyz4, f4):
        return self.dec(xyz, xyz1, f1, xyz2, f2, xyz3, f3, xyz4, f4)

    def loss(self, pred, target):
        return F.cross_entropy(pred, target)