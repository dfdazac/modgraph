import subprocess
import os

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot
import numpy as np
import scipy.sparse as sp

from utils import adj_from_edge_index
from g2g.model import Graph2Gauss
from g2g.utils import score_link_prediction

class MLPEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dims, *args):
        super(MLPEncoder, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_feat_dim, hidden_dims[0],
                                               bias=False),
                                     nn.PReLU(hidden_dims[0])])

        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i],
                                         bias=False))
            self.layers.append(nn.PReLU(hidden_dims[i]))

    def forward(self, data, edge_index, corrupt=False):
        if corrupt:
            perm = torch.randperm(data.num_nodes)
            x = data.x[perm]
        else:
            x = data.x

        z = self.layers[1](self.layers[0](x))

        for i in range(2, len(self.layers), 2):
            z = self.layers[i + 1](self.layers[i](z))

        return z


class GraphEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dims, *args):
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
    def __init__(self, input_dim, hidden_dims, encoder_class, *args):
        super(DGI, self).__init__()
        self.encoder = encoder_class(input_dim, hidden_dims)
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
    def __init__(self, input_feat_dim, hidden_dims, encoder_class, rec_weight=0):
        super(GAE, self).__init__()
        self.encoder = encoder_class(input_feat_dim, hidden_dims)
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Feature reconstruction objective using symmetric decoder
        self.rec_weight = rec_weight
        if rec_weight > 0:
            modules = []
            for i in range(len(hidden_dims) - 1, 0, -1):
                modules.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dims[0], input_feat_dim))
            modules.append(nn.ReLU())
            self.decoder = nn.Sequential(*modules)
            self.rec_loss = nn.MSELoss()
        else:
            self.rec_loss = None

    def forward(self, data, edges_pos, edges_neg):
        z = self.encoder(data, edges_pos)
        # Get scores for edges using inner product
        pos_score = (z[edges_pos[0]] * z[edges_pos[1]]).sum(dim=1)
        neg_score = (z[edges_neg[0]] * z[edges_neg[1]]).sum(dim=1)
        preds = torch.cat((pos_score, neg_score))

        targets = torch.cat((torch.ones_like(pos_score),
                             torch.zeros_like(neg_score)))
        cost = self.loss_fn(preds, targets)

        if self.rec_loss is not None:
            reconstructions = self.decoder(z)
            cost = self.rec_weight * self.rec_loss(reconstructions, data.x) + (1 - self.rec_weight) * cost

        return cost


class LookupEncoder(nn.Module):
    """Dummy encoder class to store embeddings from some algorithms,
    e.g. node2vec, as a lookup table.
    Args:
        - embeddings: (N, D) tensor, N is the number of nodes, D the dimension
            of the embeddings
    """
    def __init__(self, embeddings):
        super(LookupEncoder, self).__init__()
        self.embeddings = nn.Parameter(embeddings)

    def forward(self, data, edge_index, *args, **kwargs):
        return self.embeddings


class Node2Vec(nn.Module):
    node2vec_path = 'node2vec/main.py'

    def __init__(self, edge_index, path, num_nodes, dim=128):
        super(Node2Vec, self).__init__()
        # Write edge list
        edges_path = path + '.edges'
        embs_path = path + '.emb'
        np.savetxt(edges_path, edge_index.cpu().numpy().T, fmt='%d %d')

        subprocess.run(['python',
                        self.node2vec_path,
                        '--input', edges_path,
                        '--output', embs_path,
                        '--dimensions', str(dim),
                        '--workers', '16'])

        # Read embeddings and store in encoder
        emb_data = np.loadtxt(embs_path, skiprows=1)
        os.remove(edges_path)
        os.remove(embs_path)
        embeddings = np.zeros([num_nodes, emb_data.shape[1] - 1])
        idx = emb_data[:, 0].astype(np.int)
        embeddings[idx] = emb_data[:, 1:]
        all_embs = torch.tensor(embeddings, dtype=torch.float32)
        self.encoder = LookupEncoder(all_embs)


class G2G(nn.Module):
    def __init__(self, data, n_hidden, dim, K, train_ones, val_ones, val_zeros,
                 test_ones, test_zeros, epochs, lr, link_prediction):
        super(G2G, self).__init__()

        train_ones = train_ones.cpu().numpy().T

        A = adj_from_edge_index(data.edge_index)
        X = sp.csr_matrix(data.x.cpu().numpy())

        if link_prediction:
            val_ones = val_ones.cpu().numpy().T
            val_zeros = val_zeros.cpu().numpy().T
            test_ones = test_ones.cpu().numpy().T
            test_zeros = test_zeros.cpu().numpy().T

            g2g = Graph2Gauss(A, X, dim, train_ones, val_ones, val_zeros,
                              test_ones, test_zeros, K, n_hidden=n_hidden,
                              max_iter=epochs, lr=lr)
        else:
            g2g = Graph2Gauss(A, X, dim, train_ones, val_ones=None,
                              val_zeros=None, test_ones=None, test_zeros=None,
                              K=K, p_val=0, p_test=0, n_hidden=n_hidden,
                              max_iter=epochs, lr=lr)

        session = g2g.train()
        mu, sigma = session.run([g2g.mu, g2g.sigma])
        all_embs = torch.tensor(mu, dtype=torch.float32)
        self.encoder = LookupEncoder(all_embs)

        if link_prediction:
            test_scores = session.run(g2g.neg_test_energy, g2g.feed_dict)
            test_auc, test_ap = score_link_prediction(g2g.test_ground_truth,
                                                                test_scores)
        else:
            test_auc, test_ap = None, None

        self.test_auc, self.test_ap = test_auc, test_ap


class NodeClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super(NodeClassifier, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.4)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, data):
        x = self.dropout(self.encoder(data, data.edge_index, corrupt=False))
        x = x.detach()
        x = self.linear(x)
        return torch.log_softmax(x, dim=-1)
