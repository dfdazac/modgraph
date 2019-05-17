import os.path as osp
import itertools
from torch_geometric.datasets import Planetoid, CoraFull, Coauthor, Amazon
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score)
from sklearn.model_selection import (StratifiedShuffleSplit, GridSearchCV,
                                     ShuffleSplit, PredefinedSplit)
from sklearn.linear_model import LogisticRegressionCV
from skorch.classifier import NeuralNetClassifier, NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping
from skorch import NeuralNet


def sample_zero_entries(edge_index, seed, sample_mult=1.0):
    """Obtain zero entries from a sparse matrix.

    Args:
        edge_index (tensor): (2, N), N is the number of edges.
        seed (int): to control randomness
        sample_mult (float): the number of edges sampled is
            N * sample_mult.

    Returns:
        torch.tensor, (2, N) containing zero entries
    """
    n_edges = edge_index.shape[1]

    np.random.seed(seed)
    # Number of edges in both directions must be even
    n_samples = int(np.ceil(sample_mult * n_edges / 2) * 2)
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
            if i == n_samples:
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
    else:
        num_test = num_test//2
    if num_val is None:
        num_val = int(np.floor(edges.shape[0] / 20.))
    else:
        num_val = num_val//2

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


def link_prediction_scores(pos_score, neg_score):
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
    preds = torch.cat((pos_score, neg_score)).detach().cpu().numpy()

    targets = torch.cat((torch.ones_like(pos_score),
                         torch.zeros_like(neg_score))).cpu().numpy()

    auc_score = roc_auc_score(targets, preds)
    ap_score = average_precision_score(targets, preds)

    return auc_score, ap_score


def build_data(emb, edges_pos, edges_neg):
    # Tensors on device
    pairs_pos = torch.stack((emb[edges_pos[0]], emb[edges_pos[1]]), dim=1)
    pairs_neg = torch.stack((emb[edges_neg[0]], emb[edges_neg[1]]), dim=1)
    pairs = torch.cat((pairs_pos, pairs_neg), dim=0)

    # Tensors on CPU
    labels_pos = torch.ones(edges_pos.shape[1], dtype=torch.float32)
    labels_neg = torch.zeros(edges_neg.shape[1], dtype=torch.float32)
    labels = torch.cat((labels_pos, labels_neg), dim=-1)
    return pairs.numpy(), labels.numpy()


def train_link_prediction(score_class, emb, train_pos, train_neg,
                          test_pos, test_neg, device_str, seed=0):
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
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    emb_dim = emb.shape[1]

    print('Training link prediction model')
    early_stopping = EarlyStopping(monitor='valid_acc', threshold=1e-3,
                                   lower_is_better=False)
    net = NeuralNetBinaryClassifier(score_class, module__emb_dim=emb_dim,
                                    criterion=torch.nn.BCEWithLogitsLoss,
                                    device=device_str, max_epochs=50,
                                    verbose=1, optimizer=torch.optim.Adam,
                                    callbacks=[early_stopping],
                                    iterator_train__shuffle=True)
    params = {
        'lr': [1e-3, 5e-3, 1e-2]
    }
    # Split data into train/val
    x, targets = build_data(emb, train_pos, train_neg)
    shuffling = ShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    shuffling.get_n_splits(x, targets)
    train_index, val_index = next(shuffling.split(x, targets))
    val_fold = np.zeros(targets.shape, dtype=np.int)
    val_fold[train_index] = -1
    split = PredefinedSplit(test_fold=val_fold)
    gs = GridSearchCV(net, params, cv=split, scoring='accuracy')
    gs.fit(x, targets)

    print('Best parameters: ', gs.best_params_)
    model = gs.best_estimator_

    # Performance on train split
    preds = model.infer(x).detach().cpu()
    train_auc = roc_auc_score(targets, preds)
    train_ap = average_precision_score(targets, preds)

    # Performance on test split
    x, targets = build_data(emb, test_pos, test_neg)
    preds = model.infer(x).detach().cpu()
    test_auc = roc_auc_score(targets, preds)
    test_ap = average_precision_score(targets, preds)

    return train_auc, train_ap, test_auc, test_ap


def score_node_classification(features, targets, p_labeled=0.1, seed=0):
    """
    Train a classifier using the node embeddings as features and reports the
    performance.

    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    targets : array-like, shape [N]
        The ground truth labels
    p_labeled : float
        Percentage of nodes to use for training the classifier
    seed:

    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    lrcv = LogisticRegressionCV(cv=3, multi_class='multinomial', n_jobs=3,
                                max_iter=300, random_state=seed)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled,
                                 random_state=seed)
    split_train, split_test = next(sss.split(features, targets))
    lrcv.fit(features[split_train], targets[split_train])

    train_preds = lrcv.predict(features[split_train])
    train_acc = accuracy_score(targets[split_train], train_preds)

    test_preds = lrcv.predict(features[split_test])
    test_acc = accuracy_score(targets[split_test], test_preds)

    return train_acc, test_acc


def score_node_classification_sets(features, targets, model_class, device_str,
                                   p_labeled=0.1, seed=0):
    """
    Train a classifier using the node embeddings as features and reports the
    performance.

    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    targets : array-like, shape [N]
        The ground truth labels
    model_class: class with the definition of the classifier
    device_str: str
    p_labeled : float
        Percentage of nodes to use for training the classifier
    seed: int

    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    n_nodes, n_points, emb_dim = features.shape
    n_classes = len(np.unique(targets))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled,
                                 random_state=seed)
    split_train, split_test = next(sss.split(features, targets))

    net = NeuralNetClassifier(model_class, module__in_features=emb_dim,
                              module__n_classes=n_classes,
                              criterion=torch.nn.CrossEntropyLoss,
                              device=device_str, max_epochs=100,
                              verbose=0, optimizer=torch.optim.Adam,
                              iterator_train__shuffle=True,
                              batch_size=len(split_train))
    params = {
        'lr': [1e-3, 1e-2, 1e-1],
        'module__drop1': [0, 0.2, 0.5],
        'module__drop2': [0, 0.2, 0.5],
        'optimizer__weight_decay': [0, 1e-4, 1e-3]
    }
    gs = GridSearchCV(net, params, cv=2, scoring='accuracy')

    print('Training node classification model')
    gs.fit(features[split_train], targets[split_train])

    print('Best parameters: ', gs.best_params_)
    model = gs.best_estimator_
    predicted = model.predict(features[split_test])

    accuracy = accuracy_score(targets[split_test], predicted)

    return accuracy


def get_data(dataset_str):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset_str)

    if dataset_str in ('cora', 'citeseer', 'pubmed'):
        dataset = Planetoid(path, dataset_str)
    elif dataset_str == 'corafull':
        dataset = CoraFull(path)
    elif dataset_str == 'coauthorcs':
        dataset = Coauthor(path, name='CS')
    elif dataset_str == 'coauthorphys':
        dataset = Coauthor(path, name='Physics')
    elif dataset_str == 'amazoncomp':
        dataset = Amazon(path, name='Computers')
    elif dataset_str == 'amazonphoto':
        dataset = Amazon(path, name='Photo')
    else:
        raise ValueError(f'Unknown dataset {dataset_str}')

    return dataset[0]


def get_data_splits(dataset_str, neg_sample_mult, link_prediction,
                    add_self_connections=False, seed=0):
    data = get_data(dataset_str)
    neg_edge_index = sample_zero_entries(data.edge_index, seed,
                                         neg_sample_mult)

    if link_prediction:
        # For link prediction we split edges in train/val/test sets
        train_pos, val_pos, test_pos = split_edges(data.edge_index, seed,
                                                   add_self_connections)

        num_val, num_test = val_pos.shape[1], test_pos.shape[1]
        train_neg_all, val_neg, test_neg = split_edges(neg_edge_index, seed,
                                                       num_val=num_val,
                                                       num_test=num_test)
    else:
        train_pos, val_pos, test_pos = data.edge_index, None, None
        train_neg_all, val_neg, test_neg = neg_edge_index, None, None

    pos_split = [train_pos, val_pos, test_pos]
    neg_split = [train_neg_all, val_neg, test_neg]

    return data, pos_split, neg_split


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
    Form all valid triplets (pairwise constraints) from a set of sampled nodes
    in triplets

    Parameters
    ----------
    sampled_hops : array-like, shape [N, K]
       The sampled nodes.
    scale_terms : dict
        The appropriate up-scaling terms to ensure unbiased estimates for each
        neighbourhood

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


def sample_triplets(hops):
    return to_triplets(sample_all_hops(hops))
