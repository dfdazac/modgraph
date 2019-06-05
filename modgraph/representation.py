import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from geomloss import SamplesLoss

from .sgae.von_mises_fisher import VonMisesFisher
from .sgae.hyperspherical_uniform import HypersphericalUniform


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


class HypersphericalVariational(Representation):
    def __init__(self):
        self.uniform = HypersphericalUniform(dim=128, device='cuda')

    @staticmethod
    def score(z, pairs):
        emb_dim = z.shape[1] - 1
        z_mean, z_var = torch.split(z, emb_dim, dim=1)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        z_var = F.softplus(z_var) + 1
        q_z = VonMisesFisher(z_mean, z_var)
        z = q_z.rsample()

        return EuclideanInnerProduct.score(z, pairs)

    @staticmethod
    def score_link_pred(z, pairs):
        emb_dim = z.shape[1] - 1
        z_mean, _ = torch.split(z, emb_dim, dim=1)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        return EuclideanInnerProduct.score(z_mean, pairs)

    def regularizer(self, z):
        emb_dim = z.shape[1] - 1
        z_mean, z_var = torch.split(z, emb_dim, dim=1)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        z_var = F.softplus(z_var) + 1
        q_z = VonMisesFisher(z_mean, z_var)
        kl_div = torch.distributions.kl.kl_divergence(q_z, self.uniform).mean()
        return kl_div


class PointCloud(Representation):
    def __init__(self, n_points):
        self.sinkhorn = SamplesLoss('sinkhorn', p=1, blur=0.05, diameter=10)
        self.n_points = n_points

    def score(self, z, pairs):
        num_nodes = z.shape[0]
        points = z.reshape(num_nodes, self.n_points, -1)
        nodes_x, nodes_y = pairs
        result = -self.sinkhorn(points[nodes_x], points[nodes_y])

        return result

    def score_link_pred(self, z, pairs):
        return self.score(z, pairs)


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
