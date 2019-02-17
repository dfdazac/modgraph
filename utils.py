import numpy as np
import scipy.sparse as sp
import torch


def sample_zero_entries(edge_index, seed):
    """Obtain zero entries from a sparse matrix.
    Args:
        - edge_index: tensor, (2, N), N is the number of edges.
        - n_samples: int, number of samples to obtain
    Return:
        - torch.tensor, (2, N) containing zero entries
    """
    np.random.seed(seed)
    # Number of edges in both directions must be eve
    n_samples = int(np.ceil(edge_index.shape[1]/2) * 2)
    adjacency = adj_from_edge_index(edge_index)
    zero_entries = np.zeros([2, n_samples], dtype=np.int32)
    nonzero_or_sampled = set(zip(*adjacency.nonzero()))
    i = 0
    while True:
        t = tuple(np.random.randint(0, adjacency.shape[0], 2))
        # Don't sample diagonal of the adjacency matrix
        if t[0] == t[1]:
            continue
        if t not in nonzero_or_sampled:
            # Add edge in both directions
            t_rev = (t[1], t[0])
            zero_entries[:, i] = t
            zero_entries[:, i+1] = t_rev
            i += 2
            if i >= n_samples - 1:
                break

            nonzero_or_sampled.add(t)
            nonzero_or_sampled.add(t_rev)

    return torch.tensor(zero_entries, dtype=torch.long)


def split_edges(edge_index, seed, add_self_connections=False):
    """Obtain positive and negative train/val/test edges for an *undirected*
    graph given m an edge index (as the one used in the
    torch_geometric.datasets.Planetoid class).
    Args:
        - edge_index: tensor, (2, N), N is the number of edges.
        - seed: int, use to control randomness
        - add_self_connections: bool
    """
    np.random.seed(seed)

    adj = adj_from_edge_index(edge_index)
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]),
                               shape=adj.shape)
    if add_self_connections:
        adj = adj + sp.identity(adj.shape[0])

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

    splits = []
    for edges in [train_edges, val_edges, test_edges]:
        all_edges = add_reverse_edges(torch.tensor(edges.T, dtype=torch.long))
        splits.append(all_edges)

    return splits


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


def shuffle_graph_labels(data, train_examples_per_class,
                         val_examples_per_class, seed):
    np.random.seed(seed)

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    data.val_mask = torch.zeros_like(data.train_mask)
    data.test_mask = torch.zeros_like(data.train_mask)

    labels = np.unique(data.y.cpu().numpy())
    n_train = train_examples_per_class
    n_val = val_examples_per_class
    for i in labels:
        idx = np.random.permutation(np.where(data.y == i)[0])
        data.train_mask[idx[:n_train]] = 1
        data.val_mask[idx[n_train:n_train + n_val]] = 1
        data.test_mask[idx[n_train + n_val:]] = 1

    return data
