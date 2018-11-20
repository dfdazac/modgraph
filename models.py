import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class GVAE_Encoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2):
        super(GVAE_Encoder, self).__init__()
        self.gc1 = GCNConv(input_feat_dim, hidden_dim1)
        self.gc2 = GCNConv(hidden_dim1, hidden_dim2)
        self.gc3 = GCNConv(hidden_dim1, hidden_dim2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj, return_moments=False):
        hidden1 = self.gc1(x, adj)
        mu, logvar = self.gc2(hidden1, adj), self.gc3(hidden1, adj)
        z = self.reparameterize(mu, logvar)
        if return_moments:
            return z, mu, logvar
        else:
            return z

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.encoder = GVAE_Encoder(input_feat_dim, hidden_dim1, hidden_dim2)
        self.decoder = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self, x, adj):
        z, mu, logvar = self.encoder(x, adj, return_moments=True)
        return self.decoder(z), mu, logvar

class NodeClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super(NodeClassifier, self).__init__()
        self.encoder = encoder
        self.lin = nn.Linear(hidden_dim, num_classes)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data):
        x = self.encoder(data, data.edge_index, corrupt=False)
        x = x.detach()
        x = self.lin(x)
        return torch.log_softmax(x, dim=-1)

class LinkPredictor(nn.Module):
    def __init__(self):
        super(LinkPredictor, self).__init__()

    def predict(self, emb_a, emb_b):
        score = self(emb_a, emb_b)
        return torch.sigmoid(score)

class DotLinkPredictor(LinkPredictor):
    def __init__(self, *args, **kwargs):
        super(DotLinkPredictor, self).__init__()

    def forward(self, emb_a, emb_b):
        score = torch.sum(emb_a * emb_b, dim=-1)
        return score

class BilinearLinkPredictor(LinkPredictor):
    def __init__(self, emb_dim, **hparams):
        super(BilinearLinkPredictor, self).__init__()

        dropout_rate = hparams.get('dropout_rate', 0.0)

        self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, emb_a, emb_b):
        x_a = self.dropout(emb_a)
        x_b = self.dropout(emb_b)
        score = torch.sum(torch.matmul(x_a, self.weight) * x_b, dim=-1)
        return score

class MLPLinkPredictor(LinkPredictor):
    def __init__(self, emb_dim, **hparams):
        super(MLPLinkPredictor, self).__init__()

        dropout_rate = hparams.get('dropout_rate', 0.0)
        hidden_dim = hparams.get('hidden_dim', 512)

        self.dropout = nn.Dropout(dropout_rate)
        self.linear_input = nn.Linear(2 * emb_dim, hidden_dim)
        self.linear_output = nn.Linear(hidden_dim, 1)

    def forward(self, emb_a, emb_b):
        x = torch.cat((self.dropout(emb_a), self.dropout(emb_b)), dim=-1)
        x = F.relu(self.linear_input(x))
        score = self.linear_output(x).squeeze()
        return score
