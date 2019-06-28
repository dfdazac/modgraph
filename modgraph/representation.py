import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from geomloss import SamplesLoss
from torch.distributions.kl import kl_divergence
import numpy as np

from .sgae.von_mises_fisher import VonMisesFisher
from .sgae.hyperspherical_uniform import HypersphericalUniform


class Representation:
    @staticmethod
    def score(z, pairs):
        raise NotImplementedError

    @staticmethod
    def score_link_pred(z, pairs):
        raise NotImplementedError

    @staticmethod
    def embed(z):
        return z


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

    @staticmethod
    def embed(z):
        emb_dim = z.shape[1] // 2
        mu, logsigma = torch.split(z, emb_dim, dim=1)
        return mu


class GaussianVariational(Representation):
    def __init__(self):
        self.regularizer = 0

    def score(self, z, pairs):
        # Reparameterization trick
        emb_dim = z.shape[1] // 2
        mu, logvar = torch.split(z, emb_dim, dim=1)
        z = mu + torch.randn_like(mu) * torch.sqrt(torch.exp(logvar))

        self.regularizer = 0.5 * (torch.exp(logvar) + mu ** 2 - logvar - 1)
        self.regularizer = self.regularizer.mean()

        return EuclideanInnerProduct.score(z, pairs)

    @staticmethod
    def score_link_pred(z, pairs):
        mu = GaussianVariational.embed(z)
        return EuclideanInnerProduct.score(mu, pairs)

    @staticmethod
    def embed(z):
        emb_dim = z.shape[1] // 2
        mu, logvar = torch.split(z, emb_dim, dim=1)
        return mu


class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in
    "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow
    parameters.
    """

    def __init__(self):

        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """

        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of
        u and w for invertibility will be be satisfied inside this function.
        Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """

        zk = zk.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible
        # (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return z, log_det_jacobian


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_normalized(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -(x - mean) * (x - mean)
    log_norm *= torch.reciprocal(2.*log_var.exp())
    log_norm += -0.5 * log_var
    log_norm += -0.5 * np.log(2. * np.pi)

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


class GaussianFlow(Representation, nn.Module):
    def __init__(self, in_features, num_flows=2):
        nn.Module.__init__(self)

        self.emb_dim = in_features // 2
        self.num_flows = num_flows

        self.linear_u = nn.Linear(in_features, self.num_flows * self.emb_dim)
        self.linear_w = nn.Linear(in_features, self.num_flows * self.emb_dim)
        self.linear_b = nn.Linear(in_features, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = Planar()
            self.add_module('flow_' + str(k), flow_k)

        self.regularizer = 0

    def score(self, z, pairs):
        batch_size = z.shape[0]
        u = self.linear_u(z).view(batch_size, self.num_flows, self.emb_dim, 1)
        w = self.linear_w(z).view(batch_size, self.num_flows, 1, self.emb_dim)
        b = self.linear_b(z).view(batch_size, self.num_flows, 1, 1)

        # Reparameterization trick
        emb_dim = z.shape[1] // 2
        mu, logvar = torch.split(z, emb_dim, dim=1)
        z = [mu + torch.randn_like(mu) * torch.sqrt(torch.exp(logvar))]

        # Normalizing flows
        log_det_j = 0.
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :],
                                           b[:, k, :, :])
            z.append(z_k)
            log_det_j += log_det_jacobian

        z_0 = z[0]
        z_k = z[-1]
        # ln p(z_k)  (not averaged)
        log_p_zk = log_normal_standard(z_k, dim=1)
        # ln q(z_0)  (not averaged)
        log_q_z0 = log_normal_diag(z_0, mean=mu, log_var=logvar,
                                   dim=1)
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        summed_logs = torch.sum(log_q_z0 - log_p_zk)

        # sum over batches
        summed_ldj = torch.sum(log_det_j)

        # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
        self.regularizer = (summed_logs - summed_ldj)/batch_size

        return EuclideanInnerProduct.score(z[-1], pairs)

    def score_link_pred(self, z, pairs):
        z = self.embed(z)
        return EuclideanInnerProduct.score(z, pairs)

    def embed(self, z):
        batch_size = z.shape[0]
        u = self.linear_u(z).view(batch_size, self.num_flows, self.emb_dim, 1)
        w = self.linear_w(z).view(batch_size, self.num_flows, 1, self.emb_dim)
        b = self.linear_b(z).view(batch_size, self.num_flows, 1, 1)

        emb_dim = z.shape[1] // 2
        mu, logvar = torch.split(z, emb_dim, dim=1)

        z = [mu]

        # Normalizing flows
        log_det_j = 0.
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :],
                                           b[:, k, :, :])
            z.append(z_k)
            log_det_j += log_det_jacobian

        return z[-1]


class HypersphericalVariational(Representation):
    def __init__(self):
        self.uniform = HypersphericalUniform(dim=128, device='cuda')
        self.regularizer = 0

    def score(self, z, pairs):
        emb_dim = z.shape[1] - 1
        z_mean, z_var = torch.split(z, emb_dim, dim=1)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        z_var = F.softplus(z_var) + 1
        q_z = VonMisesFisher(z_mean, z_var)
        z = q_z.rsample()

        self.regularizer = kl_divergence(q_z, self.uniform).mean()

        return EuclideanInnerProduct.score(z, pairs)

    @staticmethod
    def score_link_pred(z, pairs):
        z_mean = HypersphericalVariational.embed(z)
        return EuclideanInnerProduct.score(z_mean, pairs)

    @staticmethod
    def embed(z):
        emb_dim = z.shape[1] - 1
        z_mean, _ = torch.split(z, emb_dim, dim=1)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        return z_mean


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
