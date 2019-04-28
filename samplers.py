import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import adj_from_edge_index, get_hops, sample_triplets


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
        triplets = sample_triplets(self.hops)
        hop_pos = np.vstack((triplets[:, 0], triplets[:, 1]))
        hop_pos = torch.tensor(hop_pos, dtype=torch.long)
        hop_neg = np.vstack((triplets[:, 0], triplets[:, 2]))
        hop_neg = torch.tensor(hop_neg, dtype=torch.long)

        return hop_pos, hop_neg


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