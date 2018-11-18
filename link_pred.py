import os.path as osp
from datetime import datetime
from argparse import ArgumentParser

import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import networkx as nx
from tensorboardX import SummaryWriter

from utils import mask_test_edges, get_roc_scores
from models import Infomax, BilinearLinkPredictor, DotLinkPredictor

def print_stats(roc_results, ap_results):
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

def build_text_summary(metadata):
    """Build a string representation of a dictionary, to be used as
    extra logging information to be read in TensorBoard
    """
    text_summary = ""
    for key, value in metadata.items():
        text_summary += '**' + str(key) + ':** ' + str(value) + '</br>'
    return text_summary

def main(model_name, n_experiments, epochs):
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

        model = model_class(emb_dim)
        adj_pred = model(emb)

        if model_name != 'dot':
            logdir = osp.join('runs', now + f'-{exper + 1:d}')
            writer = SummaryWriter(logdir)
            writer.add_text('metadata',
                            build_text_summary({'Model': model_name}))

            pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
            binary_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.Adam(model.parameters())

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

        roc_score, ap_score = get_roc_scores(adj_pred, adj_orig,
                                             val_edges, val_edges_false)
        roc_results[1, exper] = roc_score
        ap_results[1, exper] = ap_score

        roc_score, ap_score = get_roc_scores(adj_pred, adj_orig,
                                             test_edges, test_edges_false)
        roc_results[2, exper] = roc_score
        ap_results[2, exper] = ap_score

    print_stats(roc_results, ap_results)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model', help='Model name',
                        choices=['dot', 'bilinear'])
    arg_vars = vars(parser.parse_args())
    model_name = arg_vars['model']

    main(model_name, n_experiments=2, epochs=5)
