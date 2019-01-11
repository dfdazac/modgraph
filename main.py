from datetime import datetime
import os.path as osp

import torch
import numpy as np
from torch_geometric.datasets import Planetoid

from utils import split_edges, add_reverse_edges
from models import GAE, DGI

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

        print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss))

from argparse import Namespace
args = Namespace()
args.model_name = 'dgi'
args.dataset = 'cora'
args.hidden_dims = [32, 16]
args.lr = 0.001
args.epochs = 10
args.device = 'cpu'

train_encoder(args)

