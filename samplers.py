import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import adj_from_edge_index


class NodeSampling(Dataset):
    def __init__(self, iters, edge_index):
        self.edge_index = edge_index
        self.iters = torch.tensor(iters)

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        return self.iters.item()


class FirstNeighborSampling(NodeSampling):
    def __init__(self, iters, edge_index, neg_index, resample_neg):
        super(FirstNeighborSampling, self).__init__(iters, edge_index)
        self.neg_index = neg_index
        self.resample_neg = resample_neg

    def __getitem__(self, item):
        if self.resample_neg:
            num_samples = self.edge_index.shape[1]
            num_neg = self.neg_index.shape[1]
            np.random.seed(item)
            rand_idx = np.random.choice(np.arange(num_neg), num_samples,
                                        replace=False)

            neg_index = self.neg_index[:, rand_idx]
        else:
            neg_index = self.neg_index

        return self.edge_index, neg_index


class RankedSampling(NodeSampling):
    def __init__(self, iters, edge_index):
        super(RankedSampling, self).__init__(iters, edge_index)
        adj = adj_from_edge_index(edge_index)
        self.hops = get_hops(adj)

    def __getitem__(self, item):
        triplets = to_triplets(sample_all_hops(self.hops))
        hop_pos = np.vstack((triplets[:, 0], triplets[:, 1]))
        hop_pos = torch.tensor(hop_pos, dtype=torch.long)
        hop_neg = np.vstack((triplets[:, 0], triplets[:, 2]))
        hop_neg = torch.tensor(hop_neg, dtype=torch.long)

        return self.edge_index, hop_pos, hop_neg


def get_hops(A, K=1):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The graph represented as a sparse matrix
    K : int
        The maximum hopness to consider.

    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse
        matrices
    """
    hops = {1: A.tolil(), -1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop
        hops[-1] += next_hop

    return hops


def to_triplets(sampled_hops):
    """
    Form all valid triplets (pairwise constraints) from a set of sampled nodes in triplets

    Parameters
    ----------
    sampled_hops : array-like, shape [N, K]
       The sampled nodes.
    scale_terms : dict
        The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood

    Returns
    -------
    triplets : array-like, shape [?, 3]
       The transformed triplets.
    """
    triplets = []

    for i, j in itertools.combinations(np.arange(1, sampled_hops.shape[1]), 2):
        triplet = sampled_hops[:, [0] + [i, j]]
        triplet = triplet[(triplet[:, 1] != -1) & (triplet[:, 2] != -1)]
        triplet = triplet[(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])]
        triplets.append(triplet)

    return np.row_stack(triplets)


def sample_last_hop(A, nodes):
    """
    For each node in nodes samples a single node from their last (K-th)
    neighborhood.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix encoding which nodes belong to any of the 1, 2, ..., K-1,
        neighborhoods of every node
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N]
        The sampled nodes.
    """
    N = A.shape[0]

    # Sample any nodes at random
    sampled = np.random.randint(0, N, len(nodes))

    # For all rows, select sampled columns.
    # Count how many entries are nonzero in any of the 1, ..., K neighborhoods
    nnz = A[nodes, sampled].nonzero()[1]
    while len(nnz) != 0:
        # If there are still nonzero entries, generate samples again
        # BUT ONLY for those that were nonzero before (hence size=len(nnz))
        new_sample = np.random.randint(0, N, len(nnz))
        # Replace those samples that generated nonzero entries
        sampled[nnz] = new_sample
        # Count nonzeros again
        nnz = A[nnz, new_sample].nonzero()[1]

    # The returned sampled nodes are neighbors that are not in any of the
    # the 1, 2, ... K neighborhoods
    return sampled


def sample_all_hops(hops, nodes=None):
    """
    For each node in nodes samples a single node from all of their
    neighborhoods.

    Parameters
    ----------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse
        matrices
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N, K]
        The sampled nodes.
    """

    N = hops[1].shape[0]

    if nodes is None:
        nodes = np.arange(N)

    return np.vstack((nodes,
                      np.array([[-1 if len(x) == 0 else np.random.choice(x) for x in hops[h].rows[nodes]]
                                for h in hops.keys() if h != -1]),
                      sample_last_hop(hops[-1], nodes)
                      )).T


class GraphCorruptionSampling(NodeSampling):
    def __init__(self, iters, edge_index, num_nodes):
        super(GraphCorruptionSampling, self).__init__(iters, edge_index)
        self.num_nodes = num_nodes

    def __getitem__(self, item):
        perm = torch.randperm(self.num_nodes)
        return self.edge_index, perm[self.edge_index]


def simple_collate_fn(batch):
    return batch[0]


def make_sample_iterator(sampling, num_workers):
    return iter(DataLoader(sampling, batch_size=1, shuffle=False,
                           num_workers=num_workers,
                           collate_fn=simple_collate_fn))