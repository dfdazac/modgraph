import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot


class Representation:
    @staticmethod
    def score(z, pairs):
        raise NotImplementedError

    @staticmethod
    def score_link_pred(z, pairs):
        raise NotImplementedError


class EuclideanInnerProduct(Representation):
    """Euclidean representation with an inner product score"""
    @staticmethod
    def score(z, pairs):
        result = (z[pairs[0]] * z[pairs[1]]).sum(dim=1)
        return result

    @staticmethod
    def score_link_pred(z, pairs):
        return EuclideanInnerProduct.score(z, pairs)


class EuclideanInfomax(Representation, nn.Module):
    """Euclidean representation with a parameterized bilinear product score
    with a summary representation
    """
    def __init__(self, in_features):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.Tensor(in_features, in_features))
        glorot(self.weight)

    def score(self, summary, z):
        result = torch.matmul(z, torch.matmul(self.weight, summary))
        return result

    def forward(self, summary, z):
        return self.score(summary, z)

    @staticmethod
    def score_link_pred(z, pairs):
        return EuclideanInnerProduct.score(z, pairs)


class EuclideanDistance(Representation):
    """Euclidean representation with an L2 distance score"""
    @staticmethod
    def score(z, pairs):
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
        nodes_x, nodes_y = pairs
        mu_x, mu_y = z[nodes_x], z[nodes_y]
        dist = torch.sum((mu_x - mu_y) ** 2, dim=1)

        return -dist

    @staticmethod
    def score_link_pred(z, pairs):
        return EuclideanDistance.score(z, pairs)


class Gaussian(Representation):
    """Gaussian distribution representation with a -KL divergence score"""
    @staticmethod
    def score(z, pairs):
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
        emb_dim = z.shape[1] // 2
        mu, logsigma = torch.split(z, emb_dim, dim=1)
        sigma = F.elu(logsigma) + 1 + 1e-14
        nodes_x, nodes_y = pairs
        L = mu.shape[1]
        mu_x, mu_y = mu[nodes_x], mu[nodes_y]
        sigma_x, sigma_y = sigma[nodes_x], sigma[nodes_y]

        sigma_ratio = sigma_y / sigma_x
        trace_fac = sigma_ratio.sum(dim=1)
        log_det = torch.log(sigma_ratio + 1e-14).sum(dim=1)

        mu_diff_sq = (mu_x - mu_y)**2 / sigma_x
        mu_diff_sq = mu_diff_sq.sum(dim=1)

        return -0.5 * (trace_fac + mu_diff_sq - L - log_det)

    @staticmethod
    def score_link_pred(z, pairs):
        return Gaussian.score(z, pairs)


class GaussianVariational(Representation):
    @staticmethod
    def score(z, pairs):
        # Reparameterization trick
        emb_dim = z.shape[1] // 2
        mu, logvar = torch.split(z, emb_dim, dim=1)
        z = mu + torch.randn_like(mu) * torch.sqrt(torch.exp(logvar))

        return EuclideanInnerProduct.score(z, pairs)

    @staticmethod
    def score_link_pred(z, pairs):
        emb_dim = z.shape[1] // 2
        mu, _ = torch.split(z, emb_dim, dim=1)
        return EuclideanInnerProduct.score(mu, pairs)

    @staticmethod
    def regularizer(z):
        emb_dim = z.shape[1] // 2
        mu, logvar = torch.split(z, emb_dim, dim=1)
        kl_div = 0.5 * (torch.exp(logvar) + mu ** 2 - logvar - 1).mean()
        return kl_div
