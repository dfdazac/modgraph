import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from node2vec import node2vec
from gensim.models import Word2Vec
from utils import adj_from_edge_index

from representation import EuclideanBilinear


class EmbeddingMethod(nn.Module):
    def __init__(self, encoder, representation, loss, sampling_class):
        super(EmbeddingMethod, self).__init__()

        self.encoder = encoder
        self.representation = representation
        self.loss = loss
        self.sampling_class = sampling_class

    def score_pairs(self, x, edge_index, pairs):
        z = self.encoder(x, edge_index)
        return self.representation.score_link_pred(z, pairs)

    def forward(self, x, edge_index, pos_samples, neg_samples):
        if isinstance(self.representation, EuclideanBilinear):
            pos_samples = self.encoder(x, pos_samples)
            neg_samples = self.encoder(x, neg_samples)
            summary = torch.sigmoid(pos_samples.mean(dim=0))
            z = summary
        else:
            z = self.encoder(x, edge_index)

        pos_score = self.representation.score(z, pos_samples)
        neg_score = self.representation.score(z, neg_samples)
        return self.loss(pos_score, neg_score)

    def __repr__(self):
        repr_str = 'EmbeddingMethod(\n' + \
                   'Encoder: ' + self.encoder.__repr__() + \
                   '\nRepresentation: ' + self.representation.__class__.__name__ + \
                   '\nLoss: ' + self.loss.__name__ + \
                   '\nSampling: ' + self.sampling_class.__name__ + \
                   '\n)'

        return repr_str


class SGE(nn.Module):
    def __init__(self, encoder, emb_dim=None, n_points=None, **kwargs):
        super(SGE, self).__init__()
        self.encoder = encoder
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=1, blur=0.05,
                                    diameter=10)
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
