import os.path as osp
from argparse import ArgumentParser

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch
import numpy as np

from models import NodeClassifier, Infomax
from utils import adj_from_edge_index, split_edges

parser = ArgumentParser()
parser.add_argument('dataset', choices=['cora', 'citeseer', 'pubmed'])
arg_vars = vars(parser.parse_args())
dataset = arg_vars['dataset']

def train_infomax(epoch):
    infomax.train()

    if epoch == 200:
        for param_group in infomax_optimizer.param_groups:
            param_group['lr'] = 0.0001

    infomax_optimizer.zero_grad()
    loss = infomax(data, edge_index)
    loss.backward()
    infomax_optimizer.step()
    return loss.item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
data = Planetoid(path, dataset)[0]
data = data

torch.random.manual_seed(42)
np.random.seed(42)

# Obtain edges for the link prediction task
adj = adj_from_edge_index(data.edge_index)
positive_splits, _ = split_edges(adj)
train_pos, val_pos, test_pos = positive_splits

# Add edges in reverse direction for encoding
edge_index = np.vstack((train_pos, np.flip(train_pos, axis=1)))
edge_index = torch.tensor(edge_index.T, dtype=torch.long).to(device)
data = data.to(device)

hidden_dim = 512
infomax = Infomax(data.num_features, hidden_dim).to(device)
infomax_optimizer = torch.optim.Adam(infomax.parameters(), lr=0.001)

print('Train deep graph infomax.')
epochs = 300
for epoch in range(1, epochs + 1):
    loss = train_infomax(epoch)
    print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss))

torch.save(infomax.state_dict(), osp.join('saved', f'dgi-{dataset}.p'))

classifier = NodeClassifier(infomax.encoder,
                            hidden_dim,
                            data.num_classes).to(device)

classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

def train_classifier():
    infomax.eval()
    classifier.train()
    classifier_optimizer.zero_grad()
    output = classifier(data)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    classifier_optimizer.step()
    return loss.item()

def test_classifier():
    infomax.eval()
    classifier.eval()
    logits, accs = classifier(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

print('Train logistic regression classifier.')
for epoch in range(1, 51):
    train_classifier()
    accs = test_classifier()
    log = 'Epoch: {:02d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *accs))
