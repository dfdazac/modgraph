from datetime import datetime
import os.path as osp

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import split_edges, add_reverse_edges
from models import GAE, DGI

def eval_link_prediction(emb, edges_pos, edges_neg):
    """Evaluate the AUC and AP scores when using the provided embeddings to
    predict links between nodes.
    Args:
        - emb: tensor of shape (N, d) where N is the number of nodes and d
            the dimension of the embeddings.
        - edges_pos, edges_neg: tensors of shape (2, p) containing positive
        and negative edges, respectively, in their columns.
    Returns:
        - auc_score, float
        - ap_score, float
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

def train_encoder(args):
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    torch.random.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise ValueError(f'Device {args.device} specified '
                         'but CUDA is not available')

    device = torch.device(args.device)

    # Load data
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
    data = Planetoid(path, args.dataset)[0]
    # Obtain edges for the link prediction task
    positive_splits, negative_splits = split_edges(data.edge_index)
    train_pos, val_pos, test_pos = positive_splits
    train_neg, val_neg, test_neg = negative_splits
    # Add edges in reverse direction for encoding
    train_pos = add_reverse_edges(train_pos).to(device)
    train_neg = add_reverse_edges(train_neg).to(device)

    # Create model
    if args.model_name == 'dgi':
        model_class = DGI
    elif args.model_name == 'gae':
        model_class = GAE
    else:
        raise ValueError(f'Unknown model {args.model_name}')

    model = model_class(data.num_features, args.hidden_dims).to(device)

    # Train model
    print(f'Training {args.model_name}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = model(data, train_pos, train_neg)
        loss.backward()
        optimizer.step()

        # Evaluate
        embeddings = model.encoder(data, train_pos).detach()
        auc, ap = eval_link_prediction(embeddings, val_pos, val_neg)
        print('Epoch: {:03d}, train loss: {:.6f}, val_auc: {:6f}, val_ap: {:6f}'.format(epoch,
                                                                                        loss.item(),
                                                                                        auc, ap))

from argparse import Namespace
args = Namespace()
args.model_name = 'gae'
args.dataset = 'cora'
args.hidden_dims = [32, 16]
args.lr = 0.001
args.epochs = 200
args.device = 'cpu'

train_encoder(args)

