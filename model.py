import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.pointnet2.pointnet2_utils import three_nn
from lib.pointnet2.pytorch_utils import SharedMLP

from lib.models.attention import GNNAttention
from lib.models.encoder import Encoder
from lib.pointnet2.pointnet_modules import PointNetfeat
from lib.models.graphembeddingnetwork import GraphEmbeddingNet, GraphEncoder, GraphAggregator
from lib.models.graphmatchingnetwork import GraphMatchingNet
from torch_geometric.nn.pool import knn_graph
from lib.models.decoder import Decoder


class FEMtest(nn.Module):
    def __init__(self):
        super(FEMtest, self).__init__()
        self.encoder = Encoder()
        self._node_state_dim = 128
        self._edge_state_dim = 256
        self._node_hidden_sizes = [128]
        self._edge_hidden_sizes = [256, 256]
        self.pointfeature = PointNetfeat()
        encoder = GraphEncoder(self._node_state_dim, self._edge_state_dim, self._node_hidden_sizes, self._edge_hidden_sizes)
        aggregator = GraphAggregator(node_hidden_sizes=[64], graph_transform_sizes=[64], input_size=[32])
        self.graphMatching = GraphMatchingNet(encoder, aggregator,
                                              self._node_state_dim,
                                              self._edge_state_dim,
                                              self._node_hidden_sizes,
                                              self._edge_hidden_sizes,
                                              n_prop_layers = 2,
                                              share_prop_params=True,
                                              edge_net_init_scale=0.1,
                                              node_update_type='gru',
                                              use_reverse_direction=False,
                                              reverse_dir_param_different=False,
                                              layer_norm=False,
                                              prop_type='matching'
                                              )
        self.decoder = Decoder(num_class=3)
    def forward(self, src, tgt):
        B, N, C = src.shape

        #xyz0, emb_src = self.encoder(src)
        coord, coord1, coord_f1, coord2, coord_f2, coord3, coord_f3, coord4, coord_f4 = self.encoder(tgt)
        xyz1 = coord3
        emb_tgt = coord_f4
        xyz0 = src.cuda()
        emb_src = self.pointfeature(xyz0.transpose(2, 1))
        #emb_src = torch.ones(B,N,emb_tgt.size(2)).cuda()
        emb_src = emb_src.permute(0, 2, 1)
        emb_tgt = emb_tgt.permute(0, 2, 1)

        batch_size = xyz0.shape[0]
        num_inputs = xyz0.shape[1]
        t_xyz = xyz0.view(-1, 3)
        fd = emb_src.size()[1]
        t_f = emb_src.transpose(1, 2)
        t_feature = t_f.reshape(-1, fd)
        num_coords = xyz1.shape[1]
        s_xyz = xyz1.view(-1, 3)
        s_f = emb_tgt.transpose(1, 2)
        s_feature = s_f.reshape(-1, fd)

        f0 = torch.tensor([]).cuda()
        f1 = torch.tensor([]).cuda()
        updates1 = torch.tensor([]).cuda()
        updates2 = torch.tensor([]).cuda()
        for i in range(batch_size):
            batch = torch.cat((torch.tensor([0]*num_inputs), torch.tensor([1]*num_coords)), dim=0)
            batch = batch.cuda()
            xyz = torch.cat((t_xyz[i*num_inputs:(i+1)*num_inputs,:], s_xyz[i*num_coords:(i+1)*num_coords,:]), dim=0)

            edge_index = knn_graph(xyz, k=6, batch=batch, loop=False)
            features = torch.cat((t_feature[i*num_inputs:(i+1)*num_inputs,:], s_feature[i*num_coords:(i+1)*num_coords,:]) , dim=0)
            #features = s_feature[i*num_coords:(i+1)*num_coords,:]
            a = features[edge_index[0], :]
            b = features[edge_index[1], :]
            edge_feature = torch.cat([a, b], dim=-1)
            graph_vectors = self.graphMatching(features, edge_feature, edge_index[0], edge_index[1], batch, 2)
            update1 = graph_vectors[:num_inputs,:]
            update2 = graph_vectors[num_inputs:,:]
            update1 = torch.unsqueeze(update1,dim=0)
            updates1 = torch.cat((updates1, update1), dim=0)
            update2 = torch.unsqueeze(update2, dim=0)
            updates2 = torch.cat((updates2, update2),dim=0)
            f0 = torch.cat((f0, graph_vectors[:1, :]), dim=0)
            f1 = torch.cat((f1, graph_vectors[1:2, :]), dim=0)

        state = torch.cat((f0, f1), dim=-1)
        #print(state.shape)
        #state = state.view(B, -1)
        '''
        print("coord：", coord.shape)
        print("coord1:", coord1.shape)
        print("coord_f1", coord_f1.shape)
        print("coord2", coord2.shape)
        print("coord_f2", coord_f2.shape)
        print("coord3", coord3.shape)
        print("coord3_f3", coord_f3.shape)
        print("coord4", coord4.shape)
        print("coord_f4", coord_f4.shape)
        '''
        output = self.decoder(coord, coord1, coord_f1, coord2, coord_f2, coord3, coord_f3, coord4, updates2.transpose(2,1))
        #print(output.shape)
        #print(state.shape)
        #print("预测结果",output)

        return output, None