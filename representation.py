import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot


class Euclidean:
    """Euclidean representation with an inner product score"""
    @staticmethod
    def score(z, pairs):
        result = (z[pairs[0]] * z[pairs[1]]).sum(dim=1)
        return result

    @staticmethod
    def score_link_pred(z, pairs):
        return Euclidean.score(z, pairs)


class EuclideanBilinear(nn.Module):
    """Euclidean representation with a bilinear product score"""
    def __init__(self, in_features):
        super(EuclideanBilinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, in_features))
        glorot(self.weight)

    def score(self, summary, z):
        result = torch.matmul(z, torch.matmul(self.weight, summary))
        return result

    def forward(self, summary, z):
        return self.score(summary, z)

    @staticmethod
    def score_link_pred(z, pairs):
        return Euclidean.score(z, pairs)
