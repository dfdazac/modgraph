import argparse
import time
import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score, average_precision_score

from models import GVAE
from utils import mask_test_edges, adj_from_edge_index

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()

def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.random.manual_seed(42)
    np.random.seed(42)

    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    data = Planetoid(path, dataset)[0]
    data = data.to(device)

    adj = adj_from_edge_index(data.edge_index)
    n_nodes, feat_dim = data.x.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    # Add edges in reverse direction
    train_edges = np.vstack((train_edges, np.flip(train_edges, axis=1))).T
    train_edges = torch.tensor(train_edges, dtype=torch.long)

    pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float(
        (adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GVAE(feat_dim, args.hidden1, args.hidden2, pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        loss, mu = model(data, train_edges, adj_label, norm)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges,
                                          val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges,
                                        test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)