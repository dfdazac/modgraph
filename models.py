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

    def forward(self, data, edge_index):
        positive = self.encoder(data, edge_index, corrupt=False)
        negative = self.encoder(data, edge_index, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2

class VGEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2):
        super(VGEncoder, self).__init__()
        self.gc1 = GCNConv(input_feat_dim, hidden_dim1, bias=False)
        self.gc2 = GCNConv(hidden_dim1, hidden_dim2, bias=False)
        self.gc3 = GCNConv(hidden_dim1, hidden_dim2, bias=False)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data, edge_index, return_moments=False, **kwargs):
        hidden1 = F.relu(self.gc1(data.x, edge_index))
        mu = self.gc2(hidden1, edge_index)
        logvar = self.gc3(hidden1, edge_index)
        z = self.reparameterize(mu, logvar)
        if return_moments:
            return z, mu, logvar
        else:
            return z

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z):
        adj = torch.mm(z, z.t())
        return adj

class BilinearDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, input_dim):
        super(InnerProductDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, z):
        adj = torch.mm(torch.mm(z, self.weight), z.t())
        return adj

class VGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, pos_weight,
                 decoder='inner_product'):
        super(VGAE, self).__init__()
        self.encoder = VGEncoder(input_feat_dim, hidden_dim1, hidden_dim2)
        if decoder == 'inner_product':
            self.decoder = InnerProductDecoder()
        elif decoder == 'bilinear':
            self.decoder = BilinearDecoder(hidden_dim2)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, data, edge_index, adj_label, norm):
        z, mu, logvar = self.encoder(data, edge_index, return_moments=True)
        x = self.decoder(z)
        cost = norm * self.loss_fn(x, adj_label)

        KLD = -0.5 / data.num_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

        return cost + KLD, mu

class GraphEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2):
        super(GraphEncoder, self).__init__()
        self.gc1 = GCNConv(input_feat_dim, hidden_dim1, bias=False)
        self.gc2 = GCNConv(hidden_dim1, hidden_dim2, bias=False)

    def forward(self, data, edge_index, return_moments=False, **kwargs):
        hidden1 = F.relu(self.gc1(data.x, edge_index))
        z = self.gc2(hidden1, edge_index)
        return z

class GAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, pos_weight):
        super(GAE, self).__init__()
        self.encoder = GraphEncoder(input_feat_dim, hidden_dim1, hidden_dim2)
        self.decoder = InnerProductDecoder()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, data, edge_index, adj_label, norm):
        z = self.encoder(data, edge_index, return_moments=True)
        x = self.decoder(z)
        cost = norm * self.loss_fn(x, adj_label)

        return cost, z

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
        hidden_dim = hparams.get('hidden_dim', (512,))

        hidden_layers = [nn.Linear(2 * emb_dim, hidden_dim[0]),
                         nn.ReLU(),
                         nn.Dropout(dropout_rate)]
        for i in range(1, len(hidden_dim)):
            hidden_layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(dropout_rate))
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.linear_output = nn.Linear(hidden_dim[-1], 1)

    def forward(self, emb_a, emb_b):
        x = torch.cat((emb_a, emb_b), dim=-1)
        x = self.hidden_layers(x)
        score = self.linear_output(x).squeeze()
        return score
