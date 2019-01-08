from datetime import datetime
import os.path as osp

import torch
import numpy as np
from torch_geometric.datasets import Planetoid

from utils import adj_from_edge_index, split_edges

def train_encoder(model_name, dataset):
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    torch.random.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    data = Planetoid(path, dataset)[0]
    # Obtain edges for the link prediction task
    positive_splits, negative_splits = split_edges(data.edge_index)
    train_pos, val_pos, test_pos = positive_splits
    train_neg, val_neg, test_neg = negative_splits

train_encoder('gae', 'cora')

