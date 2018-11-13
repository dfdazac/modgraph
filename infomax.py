import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocessing import mask_test_edges

hidden_dim = 512

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
data = Planetoid(path, dataset)[0]


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()
        self.conv = GCNConv(data.num_features, hidden_dim)
        self.prelu = nn.PReLU(hidden_dim)

    def forward(self, x, edge_index, corrupt=False):
        if corrupt:
            perm = torch.randperm(data.num_nodes)
            x = x[perm]

        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        x = torch.matmul(x, torch.matmul(self.weight, summary))
        return x


class Infomax(nn.Module):
    def __init__(self, hidden_dim):
        super(Infomax, self).__init__()
        self.encoder = Encoder(hidden_dim)
        self.discriminator = Discriminator(hidden_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        positive = self.encoder(x, edge_index, corrupt=False)
        negative = self.encoder(x, edge_index, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
infomax = Infomax(hidden_dim).to(device)
infomax_optimizer = torch.optim.Adam(infomax.parameters(), lr=0.001)


def train_infomax(epoch):
    infomax.train()

    if epoch == 200:
        for param_group in infomax_optimizer.param_groups:
            param_group['lr'] = 0.0001

    infomax_optimizer.zero_grad()
    loss = infomax(data.x, data.edge_index)
    loss.backward()
    infomax_optimizer.step()
    return loss.item()

print('Train deep graph infomax.')
epochs = 300
for epoch in range(1, epochs + 1):
    loss = train_infomax(epoch)
    print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss))

torch.save(infomax.state_dict(), 'dgi.p')
