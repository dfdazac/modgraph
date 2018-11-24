import os.path as osp
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import numpy as np
import networkx as nx
from tensorboardX import SummaryWriter
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import mask_test_edges, get_roc_scores, split_edges
from models import Infomax, DotLinkPredictor, BilinearLinkPredictor,\
    MLPLinkPredictor

def log_stats(roc_results, ap_results, logdir, metadata_dict):
    writer = SummaryWriter(logdir)
    splits = ['train', 'val', 'test']

    for i in range(len(splits)):
        split = splits[i]
        print(f'{split} results:')
        roc_mean = np.mean(roc_results[i]) * 100
        roc_std = np.std(roc_results[i]) * 100
        ap_mean = np.mean(ap_results[i]) * 100
        ap_std = np.std(ap_results[i]) * 100
        print('\troc = {:.2f} ± {:.2f}, ap = {:.2f} ± {:.2f}'.format(roc_mean,
            roc_std, ap_mean, ap_std))

        writer.add_scalar(f'all/{split}/auc_mean', roc_mean)
        writer.add_scalar(f'all/{split}/auc_std', roc_std)
        writer.add_scalar(f'all/{split}/ap_mean', ap_mean)
        writer.add_scalar(f'all/{split}/ap_std', ap_std)

        results = {'AUC': f'{roc_mean:.2f} ± {roc_std:.2f}',
                   'AP': f'{ap_mean:.2f} ± {ap_std:.2f}'}
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

def eval_scores(model, node_embeddings, pos_edges, neg_edges, device):
    model.eval()
    edges_test = torch.tensor(np.vstack((pos_edges, neg_edges)),
                              dtype=torch.long)
    # Get node features and edge labels
    x_test = node_embeddings(edges_test).to(device)
    y_test = np.concatenate((np.ones(pos_edges.shape[0]),
                             np.zeros(neg_edges.shape[0])))
    # Predict
    y_pred = model.predict(x_test[:, 0], x_test[:, 1]).detach().cpu().numpy()

    auc_score = roc_auc_score(y_test, y_pred)
    ap_score = average_precision_score(y_test, y_pred)

    return auc_score, ap_score

def train(model_name, n_experiments, epochs, **hparams):
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    metadata_dict = {**{'Model': model_name}, **hparams}

    print(f'Link prediction model: {model_name}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'bilinear':
        model_class = BilinearLinkPredictor
    elif model_name == 'dot':
        model_class = DotLinkPredictor
    elif model_name in ['mlp', 'mlp2']:
        model_class = MLPLinkPredictor
    else:
        raise ValueError(f'Invalid model name {model_name}')

    # Load data
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    data = Planetoid(path, dataset)[0]

    # Load pretrained DGI model
    emb_dim = 512
    infomax = Infomax(data.num_features, emb_dim)
    infomax.load_state_dict(torch.load(osp.join('saved', 'dgi.p')))
    infomax.eval()

    roc_results = np.empty([3, n_experiments], dtype=np.float)
    ap_results = np.empty([3, n_experiments], dtype=np.float)

    for exper in range(n_experiments):
        print('Experiment {:d}'.format(exper + 1))
        # Obtain edges for the link prediction task
        adj = nx.adjacency_matrix(nx.from_edgelist(data.edge_index.numpy().T))
        positive_splits, negative_splits = split_edges(adj)
        train_pos, val_pos, test_pos = positive_splits
        train_neg, val_neg, test_neg = negative_splits

        # Encode graph and create a lookup table for node embeddings
        emb = infomax.encoder(data,
                              torch.tensor(train_pos.T, dtype=torch.long))
        node_embeddings = nn.Embedding(*emb.shape, _weight=emb)
        node_embeddings.weight.requires_grad = False

        edges_train = torch.tensor(np.vstack((train_pos, train_neg)),
                                   dtype=torch.long)
        x_train = node_embeddings(edges_train).to(device)
        y_train = torch.cat((torch.ones(train_pos.shape[0]),
                             torch.zeros(train_neg.shape[0]))).to(device)

        model = model_class(emb_dim, **hparams).to(device)

        if model_name != 'dot':
            # Write model name and hyperparameters to log
            logdir = osp.join('runs', f'{model_name}-{now}-{exper + 1:d}')
            writer = SummaryWriter(logdir)
            writer.add_text('metadata', build_text_summary(metadata_dict))

            binary_loss = torch.nn.BCEWithLogitsLoss().to(device)
            learning_rate = hparams.get('learning_rate', 1e-3)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)

            for epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad()
                y_pred = model(x_train[:, 0], x_train[:, 1])
                loss = binary_loss(y_pred, y_train)
                loss.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    auc_score, ap_score = eval_scores(model, node_embeddings,
                                                      train_pos, train_neg, device)
                    writer.add_scalar('train/loss', loss.item(), epoch)
                    writer.add_scalar('train/auc', auc_score, epoch)
                    writer.add_scalar('train/ap', ap_score, epoch)

                    auc_score, ap_score = eval_scores(model, node_embeddings,
                                                      val_pos, val_neg, device)
                    log = '\rEpoch {:03d}/{:03d} loss = {:.4f} val_roc = {:.3f} val_ap = {:.3f}'
                    print(log.format(epoch, epochs, loss.item(), auc_score, ap_score), end='',
                          flush=True)

                    writer.add_scalar('valid/auc', auc_score, epoch)
                    writer.add_scalar('valid/ap', ap_score, epoch)

            print()
            writer.close()

        # Train
        auc_score, ap_score = eval_scores(model, node_embeddings,
                                          train_pos, train_neg, device)
        roc_results[0, exper] = auc_score
        ap_results[0, exper] = ap_score

        # Validation
        auc_score, ap_score = eval_scores(model, node_embeddings,
                                          val_pos, val_neg, device)
        roc_results[1, exper] = auc_score
        ap_results[1, exper] = ap_score

        # Test
        auc_score, ap_score = eval_scores(model, node_embeddings,
                                          test_pos, test_neg, device)
        roc_results[2, exper] = auc_score
        ap_results[2, exper] = ap_score

    logdir = osp.join('runs', f'{model_name}-{now}-all')
    log_stats(roc_results, ap_results, logdir, metadata_dict)

def hparam_search(model_name, n_experiments, epochs):
    param_grid = {'learning_rate': [1e-4, 1e-3, 1e-2],
                  'dropout_rate': [0.1, 0.25, 0.5]}

    if model_name == 'mlp':
        param_grid['hidden_dim'] = [(700,), (512)]
    elif model_name == 'mlp2':
        param_grid['hidden_dim'] = [(800, 600), (700, 500), (700, 300)]

    grid = ParameterGrid(param_grid)

    for i, hparams in enumerate(grid):
        print(f'Hyperparameters setting {i + 1:d}/{len(grid):d}')
        train(model_name, n_experiments, epochs, **hparams)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model', help='Model name',
                        choices=['dot', 'bilinear', 'mlp', 'mlp2'])
    parser.add_argument('--search', '-s', dest='search', action='store_true',
                        help='Set to search hyperparameters for the model')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs for training')
    parser.add_argument('--nexp', type=int, default=1,
                        help='Number of experiments to run with random splits')

    arg_vars = vars(parser.parse_args())
    model_name = arg_vars['model']
    search = arg_vars['search']
    epochs = arg_vars['epochs']
    n_experiments = arg_vars['nexp']

    torch.random.manual_seed(42)
    np.random.seed(42)

    if search:
        hparam_search(model_name, n_experiments, epochs)
    else:
        train(model_name, n_experiments, epochs)
