import subprocess
import os

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, MessagePassing
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
                                     nn.ReLU(hidden_dims[0])])

        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i],
                                         bias=False))
            # FIXME
            #self.layers.append(nn.ReLU(hidden_dims[i]))

    def forward(self, data, edge_index, corrupt=False):
        if corrupt:
            perm = torch.randperm(data.num_nodes)
            x = data.x[perm]
        else:
            x = data.x

        z = self.layers[1](self.layers[0](x))

        for i in range(2, len(self.layers), 2):
            # FIXME
            #z = self.layers[i + 1](self.layers[i](z))
            z = self.layers[i](z)

        return z


class GCNEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dims, *args):
        super(GCNEncoder, self).__init__()

        self.layers = nn.ModuleList([GCNConv(input_feat_dim, hidden_dims[0],
                                             bias=False),
                                     nn.ReLU()])

        for i in range(1, len(hidden_dims)):
            self.layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i],
                                       bias=False))
            # FIXME
            #self.layers.append(nn.ReLU())

    def forward(self, data, edge_index, corrupt=False):
        if corrupt:
            perm = torch.randperm(data.num_nodes)
            x = data.x[perm]
        else:
            x = data.x

        z = self.layers[1](self.layers[0](x, edge_index))

        for i in range(2, len(self.layers), 2):
            # FIXME
            #z = self.layers[i + 1](self.layers[i](z, edge_index))
            z = self.layers[i](z, edge_index)

        return z


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


class SGCEncoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dims, *args):
        super(SGCEncoder, self).__init__()

        out_channels = hidden_dims[-1]
        K = len(hidden_dims)
        self.layer = SGConv(input_feat_dim, out_channels, K, cached=True,
                            bias=False)

    def forward(self, data, edge_index, corrupt=False):
        if corrupt:
            perm = torch.randperm(data.num_nodes)
            x = data.x[perm]
        else:
            x = data.x

        z = self.layer(x, edge_index)

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
    def __init__(self, encoder, emb_dim, *args):
        super(DGI, self).__init__()
        self.encoder = encoder
        self.discriminator = Discriminator(emb_dim)
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
    def __init__(self, encoder, emb_dim, *args):
        super(GAE, self).__init__()
        self.encoder = encoder
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


class SGE(nn.Module):
    def __init__(self, encoder, emb_dim, n_points, *args):
        super(SGE, self).__init__()
        self.encoder = encoder
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=10)
        self.space_dim = emb_dim//n_points
        self.n_points = n_points

    def score_pairs(self, embs, nodes_x, nodes_y):
        return -self.sinkhorn(embs[nodes_x].reshape(-1, self.n_points, self.space_dim),
                             embs[nodes_y].reshape(-1, self.n_points, self.space_dim))

    def forward(self, data, edges_pos, edges_neg):
        z = self.encoder(data, edges_pos)
        pos_energy = self.score_pairs(z, edges_pos[0], edges_pos[1])
        neg_energy = self.score_pairs(z, edges_neg[0], edges_neg[1])

        loss = (pos_energy**2 + torch.exp(neg_energy)).mean()

        return loss


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        device = x.device
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points)
        mu = mu.to(device)
        nu = nu.to(device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C


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
    def __init__(self, data, encoder, n_hidden, dim, train_ones, val_ones,
                 val_zeros, test_ones, test_zeros, epochs, lr, K,
                 link_prediction, energy='sqeuclidean'):
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
                              max_iter=epochs, lr=lr, encoder=encoder,
                              energy=energy)
        else:
            g2g = Graph2Gauss(A, X, dim, train_ones, val_ones=None,
                              val_zeros=None, test_ones=None, test_zeros=None,
                              K=K, p_val=0, p_test=0, n_hidden=n_hidden,
                              max_iter=epochs, lr=lr, encoder=encoder,
                              energy=energy)

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


class InnerProductScore(nn.Module):
    def __init__(self, *args, **kwargs):
        super(InnerProductScore, self).__init__()
        # For compatibility with Skorch
        self._ = nn.Parameter()

    def forward(self, emb):
        emb_a, emb_b = emb[:, 0], emb[:, 1]
        score = torch.sum(emb_a * emb_b, dim=-1)
        return score


class BilinearScore(nn.Module):
    def __init__(self, emb_dim):
        super(BilinearScore, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.dropout = nn.Dropout(0.3)

    def forward(self, emb):
        emb_a, emb_b = emb[:, 0], emb[:, 1]
        x_a = self.dropout(emb_a)
        x_b = self.dropout(emb_b)
        score = torch.sum(torch.matmul(x_a, self.weight) * x_b, dim=-1)
        return score


class DeepSetClassifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super(DeepSetClassifier, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(in_features, in_features),
                                 nn.ReLU(),
                                 nn.Linear(in_features, in_features),
                                 nn.ReLU())
        self.linear_out = nn.Linear(in_features, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.mlp(x)
        out, _ = torch.max(out, dim=1)
        out = self.dropout(out)
        out = self.linear_out(out)
        out = self.dropout(out)

        return out