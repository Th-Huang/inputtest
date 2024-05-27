import torch

from torch_geometric.nn import knn_graph, knn, CGConv
from lib.models.graphembeddingnetwork import GraphEmbeddingNet


class GNNAttention(torch.nn.Module):
    '''Uses 2 graph layers. One for self attention and one for cross attention. Self-attention based on k-NN of coordinates. Cross-attention based on k-NN in feature space'''

    def __init__(self, dim, k_input, k_coord):
        '''dim is the feature dimensions, k is the number of neighbours to consider'''
        super().__init__()

        self.k_input = k_input
        self.k_coord = k_coord
        self.conv1 = CGConv(dim, aggr='max', batch_norm=True).cuda()
        self.conv2 = CGConv(dim, aggr='max', batch_norm=True).cuda()
        self.cross_conv = GraphEmbeddingNet(node_state_dim=32,
                                            edge_state_dim=16,
                                            edge_hidden_sizes=[64,64],
                                            node_hidden_sizes=[64],
                                            n_prop_layers=5)

    def forward(self, input_xyz, coord_xyz, input_f, coord_f):
        b_i, npoints_i, d_i = input_f.shape
        input_batch_idx = torch.arange(b_i).repeat_interleave(npoints_i).to(input_xyz.device)
        f0 = input_f.reshape(-1, d_i)

        b_c, npoints_c, d_c = coord_f.shape
        coord_batch_idx = torch.arange(b_c).repeat_interleave(npoints_c).to(coord_xyz.device)
        f1 = coord_f.reshape(-1, d_c)

        #creates edge graph for coordinates
        edge_idx_input = knn_graph(input_xyz.reshape(-1, 3), k=self.k_input, batch=input_batch_idx)
        edge_idx_coord = knn_graph(coord_xyz.reshape(-1, 3), k=self.k_coord, batch=coord_batch_idx)
        #self-attention
        f0 = self.conv1(f0, edge_idx_input)
        f1 = self.conv1(f1, edge_idx_coord)

        #cross_graph
        f0, f1 = self.cross_conv(f0, f175)
        return f0, f1


'''
        edge_idx_f = knn(f1, f0, k=self.k_input, batch_x=coord_batch_idx, batch_y=input_batch_idx, cosine=True)
        edge_idx_f[1] += b_i * npoints_i
        f = self.conv2(torch.cat([f0, f1], dim=0), edge_idx_f)
        f0, f1 = f[:(b_i * npoints_i)], f[(b_i * npoints_i):]

        f0 = f0.reshape(b_i, npoints_i, d_i)
        f1 = f1.reshape(b_c, npoints_c, d_c)
        return f0, f1
'''