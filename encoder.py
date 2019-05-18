import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.inits import glorot
from geomloss import SamplesLoss

from node2vec import node2vec
from gensim.models import Word2Vec
from utils import adj_from_edge_index


# Adapted from PyTorch Geometric
class SGConv(MessagePassing):
    r"""The simgple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K=1, cached=False,
                 bias=True):
        super(SGConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.cached_result = None

        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cached_result = None

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = self.lin(x)

        edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
                                        dtype=x.dtype)

        for k in range(self.K):
            x = self.propagate('add', edge_index, x=x, norm=norm)

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


class MLPEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims, *args):
        super(MLPEncoder, self).__init__()

        dims = [in_features] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(in_features=dims[i], out_features=dims[i + 1]))

        layers.append(nn.Linear(in_features=dims[-2], out_features=dims[-1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))

        return self.layers[-1](x)


class GCNEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims, *args):
        super(GCNEncoder, self).__init__()

        dims = [in_features] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(GCNConv(in_channels=dims[i], out_channels=dims[i + 1]))

        layers.append(GCNConv(in_channels=dims[-2], out_channels=dims[-1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x, edge_index))

        return self.layers[-1](x, edge_index)


class SGCEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims, *args):
        super(SGCEncoder, self).__init__()

        out_channels = hidden_dims[-1]
        K = len(hidden_dims)
        self.layer = SGConv(in_features, out_channels, K, cached=True,
                            bias=False)

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)


class GCNMLPEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims, *args):
        super(GCNMLPEncoder, self).__init__()

        self.gcn_in = GCNConv(in_channels=in_features,
                              out_channels=hidden_dims[0])
        self.linear_out = nn.Linear(in_features=hidden_dims[0],
                                    out_features=hidden_dims[1])

    def forward(self, x, edge_index):
        out = F.relu(self.gcn_in(x, edge_index))
        out = self.linear_out(out)
        return out
