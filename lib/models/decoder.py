import torch.nn as nn
import torch.nn.functional as F
from ..pointnet2.pointnet2_modules import PointNetFeaturePropagation

class Decoder(nn.Module):
    def __init__(self, num_class):
        super(Decoder, self).__init__()
        self.num_class = num_class
        #feature propagation

        self.fp4 = PointNetFeaturePropagation(128+64, [256,256])
        self.fp3 = PointNetFeaturePropagation(256+64, [256,256])
        self.fp2 = PointNetFeaturePropagation(256+64, [256,128])
        self.fp1 = PointNetFeaturePropagation(128, [128,64,64])
        self.conv1 = nn.Conv1d(64, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(32, num_class, 1)

    def forward(self, xyz, xyz1, f1, xyz2, f2, xyz3, f3, xyz4, f4):

        f3 = self.fp4(xyz3, xyz4, f3, f4)
        f2 = self.fp3(xyz2, xyz3, f2, f3)
        f1 = self.fp2(xyz1, xyz2, f1, f2)
        f0 = self.fp1(xyz, xyz1, None, f1)

        x = self.drop1(F.relu(self.bn1(self.conv1(f0))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x
