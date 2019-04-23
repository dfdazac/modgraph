import os.path as osp
from torch_geometric.datasets import Planetoid
from gnnbench import GNNBenchmark
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


def inner_product_scores(emb, edges_pos, edges_neg):
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


def score_link_prediction(score_class, emb, test_pos, test_neg,
                          device_str, train_pos=None, train_neg=None, seed=0):
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
    if train_pos is None or train_neg is None:
        model = NeuralNet(module=score_class, module__emb_dim=emb_dim,
                          criterion=torch.nn.BCEWithLogitsLoss,
                          device=device_str, batch_size=-1)
        model.initialize()
    else:
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
        X, targets = build_data(emb, train_pos, train_neg)
        shuffling = ShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        shuffling.get_n_splits(X, targets)
        train_index, val_index = next(shuffling.split(X, targets))
        val_fold = np.zeros(targets.shape, dtype=np.int)
        val_fold[train_index] = -1
        split = PredefinedSplit(test_fold=val_fold)
        gs = GridSearchCV(net, params, cv=split, scoring='accuracy')
        gs.fit(X, targets)

        print('Best parameters: ', gs.best_params_)
        model = gs.best_estimator_

    X, targets = build_data(emb, test_pos, test_neg)
    preds = model.infer(X).detach().cpu()

    auc_score = roc_auc_score(targets, preds)
    ap_score = average_precision_score(targets, preds)

    return auc_score, ap_score


def score_node_classification(features, targets, p_labeled=0.1, seed=0):
    """
    Train a classifier using the node embeddings as features and reports the performance.

    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    targets : array-like, shape [N]
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

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled,
                                 random_state=seed)
    split_train, split_test = next(sss.split(features, targets))

    lrcv.fit(features[split_train], targets[split_train])
    predicted = lrcv.predict(features[split_test])

    f1_micro = f1_score(targets[split_test], predicted, average='micro')
    f1_macro = f1_score(targets[split_test], predicted, average='macro')
    accuracy = accuracy_score(targets[split_test], predicted)

    return f1_micro, f1_macro, accuracy


def score_node_classification_sets(features, targets, model_class, device_str, p_labeled=0.1, seed=0):
    """
    Train a classifier using the node embeddings as features and reports the performance.

    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    targets : array-like, shape [N]
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
        'lr': [1e-3, 1e-2, 1e-2]
    }
    gs = GridSearchCV(net, params, cv=2, scoring='accuracy')

    print('Training link prediction model')
    gs.fit(features[split_train], targets[split_train])

    print('Best parameters: ', gs.best_params_)
    model = gs.best_estimator_
    predicted = model.predict(features[split_test])

    f1_micro = f1_score(targets[split_test], predicted, average='micro')
    f1_macro = f1_score(targets[split_test], predicted, average='macro')
    accuracy = accuracy_score(targets[split_test], predicted)

    return f1_micro, f1_macro, accuracy


def get_data(dataset_str):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset_str)

    if dataset_str in ('cora', 'citeseer', 'pubmed'):
        dataset = Planetoid(path, dataset_str)
    elif dataset_str in ('corafull', 'coauthorcs', 'coauthorphys',
                         'amazoncomp', 'amazonphoto'):
        dataset = GNNBenchmark(path, dataset_str)
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
