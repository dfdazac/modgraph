import os.path as osp
from datetime import datetime
from argparse import ArgumentParser

import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import networkx as nx
from tensorboardX import SummaryWriter
from sklearn.model_selection import ParameterGrid

from utils import mask_test_edges, get_roc_scores
from models import Infomax, BilinearLinkPredictor, DotLinkPredictor

def log_stats(roc_results, ap_results, logdir, metadata_dict):
    writer = SummaryWriter(logdir)
    splits = ['train', 'val', 'test']

    for i in [1, 2]:
        split = splits[i]
        print(f'{split} results:')
        roc_mean = np.mean(roc_results[i])
        roc_std = np.std(roc_results[i])
        ap_mean = np.mean(ap_results[i])
        ap_std = np.std(ap_results[i])
        print('\troc = {:.3f} ± {:.3f}, ap = {:.3f} ± {:.3f}'.format(roc_mean,
            roc_std, ap_mean, ap_std))

        results = {'AUC': f'{roc_mean:.3f} ± {roc_std:.3f}',
                   'AP': f'{ap_mean:.3f} ± {ap_std:.3f}'}
        metadata_dict = {**metadata_dict, **results}
        writer.add_text(f'all/{split}', build_text_summary(metadata_dict))
        writer.add_histogram(f'all/{split}/auc', roc_results)
        writer.add_histogram(f'all/{split}/ap', ap_results)

    writer.close()

def build_text_summary(metadata):
    """Build a string representation of a dictionary, to be used as
    extra logging information to be read in TensorBoard
    """
    text_summary = ""
    for key, value in metadata.items():
        text_summary += '**' + str(key) + ':** ' + str(value) + '</br>'
    return text_summary

def train(model_name, n_experiments, epochs, **hparams):
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    print(f'Link prediction model: {model_name}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'bilinear':
        model_class = BilinearLinkPredictor
    elif model_name == 'dot':
        model_class = DotLinkPredictor
    else:
        raise ValueError(f'Invalid model name {model_name}')

    # Load data
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    data = Planetoid(path, dataset)[0]
    data = data.to(device)

    # Load pretrained DGI model
    emb_dim = 512
    infomax = Infomax(data.num_features, emb_dim).to(device)
    infomax.load_state_dict(torch.load(osp.join('saved', 'dgi.p')))
    infomax.eval()

    roc_results = np.empty([3, n_experiments], dtype=np.float)
    ap_results = np.empty([3, n_experiments], dtype=np.float)

    for exper in range(n_experiments):
        print('Experiment {:d}'.format(exper + 1))
        # Obtain edges for the link prediction task
        adj = nx.adjacency_matrix(nx.from_edgelist(data.edge_index.numpy().T))
        adj_orig = adj

        # Sometimes the assertions in mask_test_edges fail, so we try again
        while True:
            try:
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
                break
            except AssertionError:
                continue

        adj_train = torch.tensor(adj_train.toarray(), dtype=torch.float32)
        train_edges = torch.tensor(train_edges, dtype=torch.long)

        # Encode graph
        emb = infomax.encoder(data, train_edges.t()).detach()

        model = model_class(emb_dim, **hparams)
        model.train()

        if model_name != 'dot':
            # Write model name and hyperparameters to log
            logdir = osp.join('runs', f'{model_name}-{now}-{exper + 1:d}')
            writer = SummaryWriter(logdir)
            metadata_dict = {**{'Model': model_name}, **hparams}
            writer.add_text('metadata', build_text_summary(metadata_dict))

            pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
            binary_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=hparams['learning_rate'])

            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()
                adj_pred = model(emb)
                loss = binary_loss(adj_pred, adj_train)
                loss.backward()
                optimizer.step()

                adj_pred = adj_pred.detach().numpy()
                roc_score, ap_score = get_roc_scores(adj_pred, adj_orig,
                                        val_edges, val_edges_false)
                log = '\rEpoch {:03d}/{:03d} loss = {:.4f} val_roc = {:.3f} val_ap = {:.3f}'
                print(log.format(epoch, epochs, loss.item(), roc_score, ap_score), end='',
                      flush=True)

                writer.add_scalar('train/loss', loss.item(), epoch)
                writer.add_scalar('valid/auc', roc_score, epoch)
                writer.add_scalar('valid/ap', ap_score, epoch)

            print()
            writer.close()

        model.eval()
        adj_pred = model(emb).detach().numpy()
        roc_score, ap_score = get_roc_scores(adj_pred, adj_orig,
                                             val_edges, val_edges_false)
        roc_results[1, exper] = roc_score
        ap_results[1, exper] = ap_score

        roc_score, ap_score = get_roc_scores(adj_pred, adj_orig,
                                             test_edges, test_edges_false)
        roc_results[2, exper] = roc_score
        ap_results[2, exper] = ap_score

    logdir = osp.join('runs', f'{model_name}-{now}-all')
    log_stats(roc_results, ap_results, logdir, metadata_dict)

def hparam_search(model_name):
    param_grid = {'learning_rate': [1e-3, 1e-2, 1e-1]}

    if model_name == 'bilinear':
        param_grid['dropout_rate'] = [0.1, 0.25, 0.5]

    grid = ParameterGrid(param_grid)

    for hparams in grid:
        train(model_name, n_experiments=10, epochs=100, **hparams)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model', help='Model name',
                        choices=['dot', 'bilinear', 'mlp'])
    parser.add_argument('-dropout_rate', '-d', type=float, default=0.0)
    parser.add_argument('--search', '-s', dest='search', action='store_true',
                        help='Set to search hyperparameters for the model')

    arg_vars = vars(parser.parse_args())
    model_name = arg_vars['model']
    search = arg_vars['search']

    if search:
        hparam_search(model_name)
    else:
        train(model_name, n_experiments=3, epochs=2, learning_rate=1e-3,
              **arg_vars)
