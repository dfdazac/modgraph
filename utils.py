import numpy as np
import scipy.sparse as sp
import torch


def sample_zero_entries(mat):
    """A generator to obtain zero entries from a sparse matrix"""
    nonzero_or_sampled = set(zip(*mat.nonzero()))
    while True:
        t = tuple(np.random.randint(0, mat.shape[0], 2))
        # Don't sample diagonal of the adjacency matrix
        if t[0] == t[1]:
            continue
        if t not in nonzero_or_sampled:
            yield t
            # Add edge in both directions
            nonzero_or_sampled.add(t)
            nonzero_or_sampled.add((t[1], t[0]))


def split_edges(edge_index):
    """Obtain positive and negative train/val/test edges for an *undirected*
    graph given m an edge index (as the one used in the
    torch_geometric.datasets.Planetoid class).
    Args:
        - edge_index: tensor, (2, N), N is the number of edges.
    """
    adj = adj_from_edge_index(edge_index)
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]),
                               shape=adj.shape)
    adj.eliminate_zeros()

    adj_triu = sp.triu(adj)
    edges = np.array(adj_triu.nonzero()).T
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    # Shuffle edges
    all_edge_idx = np.arange(edges.shape[0])
    np.random.shuffle(all_edge_idx)

    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]),
                            axis=0)

    # Sample zero entries without replacement
    zero_iterator = sample_zero_entries(adj)

    # NOTE: these edge lists only contain single direction of edge!
    positive_splits = [torch.tensor(train_edges.T, dtype=torch.long),
                       torch.tensor(val_edges.T, dtype=torch.long),
                       torch.tensor(test_edges.T, dtype=torch.long)]
    negative_splits = []

    for i in range(len(positive_splits)):
        negative_edges = np.empty(positive_splits[i].shape, dtype=np.int32)
        for j in range(negative_edges.shape[1]):
            negative_edges[:, j] = next(zero_iterator)

        negative_splits.append(torch.tensor(negative_edges, dtype=torch.long))

    return positive_splits, negative_splits


def add_reverse_edges(edges):
    """Add edges in the reverse direction to a tensor containing edges in
    one direction only.
    Args:
        - edges: tensor, (2, N), N is the number of edges.
    """
    edges_inv = torch.stack((edges[1], edges[0]), dim=0)
    all_edges = torch.cat((edges, edges_inv), dim=1)
    return all_edges


def adj_from_edge_index(edge_index):
    """Get a sparse symmetric adjacency matrix from an edge index (as the one
    used in the torch_geometric.datasets.Planetoid class).
    Args:
        - edge_index: tensor, (2, N), N is the number of edges.
    """
    n_edges = edge_index.shape[1]
    rows = edge_index[0].numpy()
    cols = edge_index[1].numpy()
    values = np.ones(n_edges, dtype=np.bool)
    partial_adj = sp.coo_matrix((values, (rows, cols)))
    boolean_adj = partial_adj + partial_adj.T
    return boolean_adj.astype(np.int64).tocsr()
