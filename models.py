import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from node2vec import node2vec
from gensim.models import Word2Vec
from utils import adj_from_edge_index

from representation import EuclideanInnerProduct, EuclideanBilinear


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
        self.representation = EuclideanInnerProduct()

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
