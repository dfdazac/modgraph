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
    def __init__(self, encoder, emb_dim=None, **kwargs):
        super(DGI, self).__init__()
        self.encoder = encoder
        self.discriminator = Discriminator(emb_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def score_pairs(self, embs, nodes_x, nodes_y):
        return (embs[nodes_x] * embs[nodes_y]).sum(dim=1)

    def forward(self, x, edge_index, edges_pos, edges_neg):
        positive = self.encoder(x, edges_pos)
        negative = self.encoder(x, edges_neg)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2


class GAE(nn.Module):
    def __init__(self, encoder, **kwargs):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.loss_fn = nn.BCEWithLogitsLoss()

    def score_pairs(self, embs, nodes_x, nodes_y):
        return (embs[nodes_x] * embs[nodes_y]).sum(dim=1)

    def forward(self, x, edge_index, edges_pos, edges_neg):
        device = next(self.parameters()).device
        edges_pos = edges_pos.to(device)
        edges_neg = edges_neg.to(device)

        z = self.encoder(x, edge_index)
        # Get scores for edges using inner product
        pos_score = (z[edges_pos[0]] * z[edges_pos[1]]).sum(dim=1)
        neg_score = (z[edges_neg[0]] * z[edges_neg[1]]).sum(dim=1)
        preds = torch.cat((pos_score, neg_score))

        targets = torch.cat((torch.ones_like(pos_score),
                             torch.zeros_like(neg_score)))
        cost = self.loss_fn(preds, targets)

        return cost


class SGE(nn.Module):
    def __init__(self, encoder, emb_dim=None, n_points=None, **kwargs):
        super(SGE, self).__init__()
        self.encoder = encoder
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=1, blur=0.05)
        self.space_dim = emb_dim//n_points
        self.n_points = n_points

    def score_pairs(self, embs, nodes_x, nodes_y):
        return -self.sinkhorn(embs[nodes_x].reshape(-1, self.n_points, self.space_dim),
                              embs[nodes_y].reshape(-1, self.n_points, self.space_dim))

    def forward(self, x, edge_index, edges_pos, edges_neg):
        z = self.encoder(x, edge_index)
        pos_energy = -self.score_pairs(z, edges_pos[0], edges_pos[1])
        neg_energy = -self.score_pairs(z, edges_neg[0], edges_neg[1])

        # BCE Loss
        # loss = (pos_energy - torch.log(1 - torch.exp(-neg_energy)) + 1e-8).mean()

        # Square exponential loss
        # loss = (pos_energy**2 + torch.exp(-neg_energy)).mean()

        # Square-square loss
        margin_diff = 1.0 - neg_energy
        margin_diff[margin_diff < 0] = 0.0
        loss = torch.mean(pos_energy**2) + torch.mean(margin_diff**2)

        # Hinge loss
        # m = 2
        # margin = m + pos_energy - neg_energy
        # margin[margin < 0] = 0
        # loss = margin.mean()

        return loss


class SGEMetric(nn.Module):
    def __init__(self, encoder, emb_dim=None, n_points=None, **kwargs):
        super(SGEMetric, self).__init__()
        self.encoder = encoder
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=1, blur=0.05)
        self.space_dim = emb_dim//n_points
        self.n_points = n_points

    def score_pairs(self, embs, nodes_x, nodes_y):
        return -self.sinkhorn(embs[nodes_x].reshape(-1, self.n_points, self.space_dim),
                              embs[nodes_y].reshape(-1, self.n_points, self.space_dim))

    def forward(self, x, edge_index, pairs, graph_dist):
        z = self.encoder(x, edge_index)
        distance = -self.score_pairs(z, pairs[0], pairs[1])
        distortion = torch.abs(distance - graph_dist)/graph_dist
        loss = torch.mean(distortion)
        return loss


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

    def forward(self, x, edge_index, *args, **kwargs):
        return self.embeddings


# noinspection PyAbstractClass
class Node2Vec(nn.Module):
    def __init__(self, edge_index, num_nodes, dim=128):
        super(Node2Vec, self).__init__()
        adj = adj_from_edge_index(edge_index)
        nx_graph = nx.from_scipy_sparse_matrix(adj)
        for edge in nx_graph.edges():
            nx_graph[edge[0]][edge[1]]['weight'] = 1
        nx_graph = nx_graph.to_undirected()

        graph = node2vec.Graph(nx_graph, is_directed=False, p=1, q=1)
        graph.preprocess_transition_probs()
        walks = graph.simulate_walks(num_walks=10, walk_length=80)
        walks = [list(map(str, walk)) for walk in walks]

        model = Word2Vec(walks, size=dim, window=10, min_count=0, sg=1,
                         workers=8, iter=1)

        embeddings = np.zeros([num_nodes, dim])
        for entity in model.wv.index2entity:
            embeddings[int(entity)] = model.wv.get_vector(entity)

        all_embs = torch.tensor(embeddings, dtype=torch.float32)
        self.encoder = LookupEncoder(all_embs)

    def score_pairs(self, embs, nodes_x, nodes_y):
        return (embs[nodes_x] * embs[nodes_y]).sum(dim=1)


class MLPGaussianEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims, *args):
        super(MLPGaussianEncoder, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=hidden_dims[0])
        self.linear_mu = nn.Linear(in_features=hidden_dims[0],
                                   out_features=hidden_dims[1])
        self.linear_logsigma = nn.Linear(in_features=hidden_dims[0],
                                         out_features=hidden_dims[1])

    def forward(self, x, edge_index):
        out = F.relu(self.linear(x))
        mu = self.linear_mu(out)
        sigma = F.elu(self.linear_logsigma(out)) + 1 + 1e-14

        return torch.stack((mu, sigma), dim=0)


class GCNGaussianEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims, *args):
        super(GCNGaussianEncoder, self).__init__()

        self.gcn_in = GCNConv(in_channels=in_features,
                              out_channels=hidden_dims[0])
        self.gcn_mu = GCNConv(in_channels=hidden_dims[0],
                              out_channels=hidden_dims[1])
        self.gcn_logsigma = GCNConv(in_channels=hidden_dims[0],
                                    out_channels=hidden_dims[1])

    def forward(self, x, edge_index):
        out = F.relu(self.gcn_in(x, edge_index))
        mu = self.gcn_mu(out, edge_index)
        sigma = F.elu(self.gcn_logsigma(out, edge_index)) + 1 + 1e-14

        return torch.stack((mu, sigma), dim=0)


class GCNMLPGaussianEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims, *args):
        super(GCNMLPGaussianEncoder, self).__init__()

        self.gcn_in = GCNConv(in_channels=in_features,
                              out_channels=hidden_dims[0])
        self.linear_mu = nn.Linear(in_features=hidden_dims[0],
                                out_features=hidden_dims[1])
        self.linear_logsigma = nn.Linear(in_features=hidden_dims[0],
                                      out_features=hidden_dims[1])

    def forward(self, x, edge_index):
        out = F.relu(self.gcn_in(x, edge_index))
        mu = self.linear_mu(out)
        sigma = F.elu(self.linear_logsigma(out)) + 1 + 1e-14

        return torch.stack((mu, sigma), dim=0)


class G2G(nn.Module):
    def __init__(self, encoder, energy=None, **kwargs):
        super(G2G, self).__init__()
        self.encoder = encoder

        if energy == 'kldiv':
            self.score_pairs = self.score_pairs_kldiv
        elif energy == 'sqeuclidean':
            self.score_pairs = self.score_pairs_sqeuclidean
        else:
            raise ValueError(f'Unknown energy {energy}')

    def score_pairs_kldiv(self, embs, nodes_x, nodes_y):
        """
        Computes the energy of a set of node pairs as the KL divergence between
        their respective Gaussian embeddings.

        Parameters
        ----------
        pairs : array-like, shape [?, 2]
            The edges/non-edges for which the energy is calculated

        Returns
        -------
        energy : array-like, shape [?]
            The energy of each pair given the currently learned model
        """
        mu, sigma = embs
        L = mu.shape[1]
        mu_x, mu_y = mu[nodes_x], mu[nodes_y]
        sigma_x, sigma_y = sigma[nodes_x], sigma[nodes_y]

        sigma_ratio = sigma_y / sigma_x
        trace_fac = sigma_ratio.sum(dim=1)
        log_det = torch.log(sigma_ratio + 1e-14).sum(dim=1)

        mu_diff_sq = (mu_x - mu_y)**2 / sigma_x
        mu_diff_sq = mu_diff_sq.sum(dim=1)

        return -0.5 * (trace_fac + mu_diff_sq - L - log_det)

    def score_pairs_sqeuclidean(self, embs, nodes_x, nodes_y):
        """
        Computes the energy of a set of node pairs as the KL divergence between
        their respective Gaussian embeddings.

        Parameters
        ----------
        pairs : array-like, shape [?, 2]
            The edges/non-edges for which the energy is calculated

        Returns
        -------
        energy : array-like, shape [?]
            The energy of each pair given the currently learned model
        """
        mu, _ = embs
        mu_x, mu_y = mu[nodes_x], mu[nodes_y]
        dist = torch.sum((mu_x - mu_y)**2, dim=1)

        return -dist

    def forward(self, x, edge_index, hop_pos, hop_neg):
        embs = self.encoder(x, edge_index)

        pos_energy = -self.score_pairs(embs, hop_pos[0], hop_pos[1])
        neg_energy = -self.score_pairs(embs, hop_neg[0], hop_neg[1])

        loss = (pos_energy ** 2 + torch.exp(-neg_energy)).mean()

        return loss


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
    def __init__(self, in_features, n_classes, drop1, drop2):
        super(DeepSetClassifier, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(in_features, in_features),
                                 nn.ReLU(),
                                 nn.Linear(in_features, in_features),
                                 nn.ReLU())
        self.linear_out = nn.Linear(in_features, n_classes)
        self.dropout1 = nn.Dropout(drop1)
        self.dropout2 = nn.Dropout(drop2)

    def forward(self, x):
        out = self.mlp(x)
        out, _ = torch.max(out, dim=1)
        out = self.dropout1(out)
        out = self.linear_out(out)
        out = self.dropout2(out)

        return out
