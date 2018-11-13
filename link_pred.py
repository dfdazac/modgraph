import os.path as osp
import math
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx

from preprocessing import mask_test_edges
from dgi import Infomax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
data = Planetoid(path, dataset)[0]
data = data.to(device)

# Load pretrained DGI model
emb_dim = 512
infomax = Infomax(data.num_features, emb_dim).to(device)
infomax.load_state_dict(torch.load('dgi.p'))

def get_roc_score(edges_pos, edges_neg, adj_pred):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_pred[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_pred[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

n_experiments = 10
dot_predictor_roc = []
dot_predictor_ap = []
bi_predictor_roc = []
bi_predictor_ap = []

for i in range(n_experiments):
    # Obtain edges for the link prediction task
    adj = nx.adjacency_matrix(nx.from_edgelist(data.edge_index.numpy().T))
    adj_orig = adj
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

    adj_train = torch.tensor(adj_train.toarray(), dtype=torch.float32)
    train_edges = torch.tensor(train_edges, dtype=torch.long)

    emb = infomax.encoder(data, train_edges.t()).detach()
    adj_pred = torch.sigmoid(torch.matmul(emb, emb.t())).numpy()

    print('Results with dot product predictor:')
    roc_score, ap_score = get_roc_score(val_edges, val_edges_false, adj_pred)
    print('val_roc = {:.3f} val_ap = {:.3f}'.format(roc_score, ap_score))

    roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_pred)
    print('test_roc = {:.3f} test_ap = {:.3f}'.format(roc_score, ap_score))

    dot_predictor_roc.append(roc_score)
    dot_predictor_ap.append(ap_score)

    class LinkPredictor(nn.Module):
        def __init__(self, emb_dim):
            super(LinkPredictor, self).__init__()

            self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)

        def forward(self, emb):
            adj_pred = torch.matmul(emb, torch.matmul(self.weight, emb.t()))
            return adj_pred

    predictor = LinkPredictor(emb_dim)
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    binary_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(predictor.parameters())

    epochs = 100
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        adj_pred = predictor(emb)
        loss = binary_loss(adj_pred, adj_train)
        loss.backward()
        optimizer.step()

        adj_pred = adj_pred.detach().numpy()
        roc_score, ap_score = get_roc_score(val_edges, val_edges_false, adj_pred)
        log = '\rEpoch: {:03d}, loss = {:.4f}, val_roc = {:.3f}, val_ap = {:.3f}'
        print(log.format(epoch, loss.item(), roc_score, ap_score), end='',
              flush=True)

    roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_pred)
    log = '\ntest_roc = {:.3f}, test_ap = {:.3f}'
    print(log.format(roc_score, ap_score))

    bi_predictor_roc.append(roc_score)
    bi_predictor_ap.append(ap_score)

def print_stats(roc_results, ap_results, name):
    log = '{}: test_roc = {:.3f} ± {:.3f}, test_ap = {:.3f} ± {:.3f}'
    print(log.format(name, 100*np.mean(roc_results), 100*np.std(roc_results),
                     100*np.mean(ap_results), 100*np.std(ap_results)))

print_stats(dot_predictor_roc, dot_predictor_ap, 'dot predictor')
print_stats(bi_predictor_roc, bi_predictor_ap, 'bilinear predictor')