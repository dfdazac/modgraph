import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot

class GraphEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dims):
        super(GraphEncoder, self).__init__()

        self.layers = nn.ModuleList([GCNConv(input_feat_dim, hidden_dims[0],
                                             bias=False),
                                     nn.PReLU(hidden_dims[0])])

        for i in range(1, len(hidden_dims)):
            self.layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i],
                                       bias=False))
            self.layers.append(nn.PReLU(hidden_dims[i]))

    def forward(self, data, edge_index, corrupt=False):
        if corrupt:
            perm = torch.randperm(data.num_nodes)
            x = data.x[perm]
        else:
            x = data.x

        z = self.layers[1](self.layers[0](x, edge_index))

        for i in range(2, len(self.layers), 2):
            z = self.layers[i + 1](self.layers[i](z, edge_index))

        return z

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        glorot(self.weight)

    def forward(self, x, summary):
        x = torch.matmul(x, torch.matmul(self.weight, summary))
        return x

class DGI(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(DGI, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dims)
        self.discriminator = Discriminator(hidden_dims[-1])
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, data, edges_pos, edges_neg):
        positive = self.encoder(data, edges_pos, corrupt=False)
        negative = self.encoder(data, edges_pos, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2

class GAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dims):
        super(GAE, self).__init__()
        self.encoder = GraphEncoder(input_feat_dim, hidden_dims)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, data, edges_pos, edges_neg):
        z = self.encoder(data, edges_pos)
        # Get scores for edges using inner product
        pos_score = (z[edges_pos[0]] * z[edges_pos[1]]).sum(dim=1)
        neg_score = (z[edges_neg[0]] * z[edges_neg[1]]).sum(dim=1)
        preds = torch.cat((pos_score, neg_score))

        targets = torch.cat((torch.ones_like(pos_score),
                             torch.zeros_like(neg_score)))
        cost = self.loss_fn(preds, targets)

        return cost

class NodeClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super(NodeClassifier, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.4)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, data):
        x = self.dropout(self.encoder(data, data.edge_index, corrupt=False))
        x = x.detach()
        x = self.linear(x)
        return torch.log_softmax(x, dim=-1)
