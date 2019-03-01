import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score)


def score_link_prediction(emb, edges_pos, edges_neg):
    """Evaluate the AUC and AP scores when using the provided embeddings to
    predict links between nodes.

    Args:
        emb: tensor of shape (N, d) where N is the number of nodes and d
            the dimension of the embeddings.
        edges_pos, edges_neg: tensors of shape (2, p) containing positive
        and negative edges, respectively, in their columns.

    Returns:
        auc_score, float
        ap_score, float
    """
    # Get scores for edges using inner product
    pos_score = (emb[edges_pos[0]] * emb[edges_pos[1]]).sum(dim=1)
    neg_score = (emb[edges_neg[0]] * emb[edges_neg[1]]).sum(dim=1)
    preds = torch.cat((pos_score, neg_score)).cpu().numpy()

    targets = torch.cat((torch.ones_like(pos_score),
                         torch.zeros_like(neg_score))).cpu().numpy()

    auc_score = roc_auc_score(targets, preds)
    ap_score = average_precision_score(targets, preds)

    return auc_score, ap_score


def sample_zero_entries(edge_index, seed, samples_fraction=1.0):
    """Obtain zero entries from a sparse matrix.

    Args:
        edge_index (tensor): (2, N), N is the number of edges.
        seed (int): to control randomness
        samples_fraction (float): the number of edges sampled is
            N * samples_fraction.

    Returns:
        torch.tensor, (2, N) containing zero entries
    """
    n_edges = edge_index.shape[1]

    np.random.seed(seed)
    # Number of edges in both directions must be even
    n_samples = int(np.ceil(samples_fraction * n_edges/2) * 2)
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


def split_edges(edge_index, seed, add_self_connections=False,
                num_val=None, num_test=None):
    """Obtain train/val/test edges for an *undirected*
    graph given m an edge index.

    Args:
        edge_index (tensor): (2, N), N is the number of edges.
        seed (int): to control randomness
        add_self_connections (bool):

    Returns:
        list, containing 3 tensors of shape (2, N) corresponding to
            train, validation and test splits respectively.
    """
    adj = adj_from_edge_index(edge_index)
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]),
                               shape=adj.shape)
    if add_self_connections:
        adj = adj + sp.identity(adj.shape[0])

    adj.eliminate_zeros()

    adj_triu = sp.triu(adj)
    edges = np.array(adj_triu.nonzero()).T
    if num_test is None:
        num_test = int(np.floor(edges.shape[0] / 10.))
    if num_val is None:
        num_val = int(np.floor(edges.shape[0] / 20.))

    # Shuffle edges
    all_edge_idx = np.arange(edges.shape[0])
    np.random.seed(seed)
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
        edges (tensor): (2, N), N is the number of edges.

    Returns:
        tensor, (2, 2*N)
    """
    edges_inv = torch.stack((edges[1], edges[0]), dim=0)
    all_edges = torch.cat((edges, edges_inv), dim=1)
    return all_edges


def adj_from_edge_index(edge_index):
    """Get a sparse symmetric adjacency matrix from an edge index (as the one
    used in the torch_geometric.datasets.Planetoid class).

    Args:
        edge_index (tensor): (2, N), N is the number of edges.

    Returns:
        scipy scr adjacency matrix
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
    """Shuffle the label masks in a data object

    Args:
        data (InMemoryDataset): object containing graph data
        train_examples_per_class (int):
        val_examples_per_class (int):
        seed (int): to control randomness

    Returns:
        InMemoryDataset with modified label masks
    """
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


def sample_edges(edge_index, n_samples, seed):
    """Sample edges at random from and edge list

    Args:
        edge_index (tensor): (2, N), N is the number of edges.
        n_samples (int): number of samples to collect
        seed (int): to control randomness

    Returns:
        list, containing 3 tensors of shape (2, N) corresponding to
            train, validation and test splits respectively.
    """
    N = edge_index.shape[1]
    np.random.seed(seed)
    rand_idx = np.random.choice(np.arange(N), n_samples, replace=False)
    return edge_index[:, rand_idx]


def score_node_classification(features, z, p_labeled=0.1, n_repeat=1, seed=0,
                              norm=False):
    """
    Train a classifier using the node embeddings as features and reports the performance.

    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    z : array-like, shape [N]
        The ground truth labels
    p_labeled : float
        Percentage of nodes to use for training the classifier
    n_repeat : int
        Number of times to repeat the experiment
    norm

    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    lrcv = LogisticRegressionCV(cv=3, multi_class='multinomial', n_jobs=-1,
                                max_iter=300, random_state=seed)

    if norm:
        features = normalize(features)

    trace = []
    for i in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled,
                                     random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        lrcv.fit(features[split_train], z[split_train])
        predicted = lrcv.predict(features[split_test])

        f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')
        accuracy = accuracy_score(z[split_test], predicted)

        trace.append((f1_micro, f1_macro, accuracy))

    return np.array(trace).mean(0)
