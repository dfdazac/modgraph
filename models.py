import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim)
        self.prelu = nn.PReLU(hidden_dim)

    def forward(self, data, edge_index, corrupt=False):
        if corrupt:
            perm = torch.randperm(data.num_nodes)
            x = data.x[perm]
        else:
            x = data.x

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
    def __init__(self, input_dim, hidden_dim):
        super(Infomax, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.discriminator = Discriminator(hidden_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, data):
        positive = self.encoder(data, data.edge_index, corrupt=False)
        negative = self.encoder(data, data.edge_index, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2

class Classifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.lin = nn.Linear(hidden_dim, num_classes)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data):
        x = self.encoder(data, data.edge_index, corrupt=False)
        x = x.detach()
        x = self.lin(x)
        return torch.log_softmax(x, dim=-1)