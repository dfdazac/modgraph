import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform

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
        uniform(size, self.weight)

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

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, data):
        x = self.encoder(data, data.edge_index, corrupt=False)
        x = x.detach()
        x = self.linear(x)
        return torch.log_softmax(x, dim=-1)

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

class BilinearDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, input_dim):
        super(BilinearDecoder, self).__init__()
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
