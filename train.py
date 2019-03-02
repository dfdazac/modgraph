import os.path as osp
import os
import time

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from gnnbench import GNNBenchmark
from sacred import Experiment
from sacred.observers import MongoObserver

from utils import (score_link_prediction, sample_zero_entries, split_edges,
                   sample_edges, score_node_classification)
from models import MLPEncoder, GCNENcoder, GAE, DGI, Node2Vec, G2G, InnerProductScore, BilinearScore


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


def train_encoder(dataset_str, method, encoder_str, dimensions, lr, epochs,
                  rec_weight, device_str, link_prediction=False, seed=0,
                  ckpt_name=None, edge_score='inner'):
    if encoder_str == 'mlp':
        encoder_class = MLPEncoder
    elif encoder_str == 'gcn':
        encoder_class = GCNENcoder
    else:
        raise ValueError(f'Unknown encoder {encoder_str}')

    if method == 'dgi':
        model_class = DGI
    elif method == 'gae':
        model_class = GAE
    elif method in ['node2vec', 'graph2gauss']:
        model_class = None
    else:
        raise ValueError(f'Unknown model {method}')

    if edge_score == 'inner':
        score_class = InnerProductScore
    elif edge_score == 'bilinear':
        score_class = BilinearScore
    else:
        raise ValueError(f'Unknown edge score {edge_score}')

    if not torch.cuda.is_available() and device_str.startswith('cuda'):
        raise ValueError(f'Device {device_str} specified '
                         'but CUDA is not available')

    device = torch.device(device_str)
    data = get_data(dataset_str)

    resample_neg_edges = False
    if not link_prediction and method == 'gae':
        resample_neg_edges = True
        neg_edge_index = sample_zero_entries(data.edge_index, seed,
                                             samples_fraction=10)
        train_pos, val_pos, test_pos = split_edges(data.edge_index, seed)

        num_val, num_test = val_pos.shape[1], test_pos.shape[1]
        train_neg_all, val_neg, test_neg = split_edges(neg_edge_index, seed,
                                                       num_val=num_val,
                                                       num_test=num_test)

        train_neg = sample_edges(train_neg_all, n_samples=train_pos.shape[1],
                                 seed=seed)
    elif link_prediction:
        neg_edge_index = sample_zero_entries(data.edge_index, seed)
        add_self_connections = method == 'node2vec'
        train_pos, val_pos, test_pos = split_edges(data.edge_index, seed,
                                                   add_self_connections)
        train_neg_all, val_neg, test_neg = split_edges(neg_edge_index, seed)
        train_neg = train_neg_all
    else:
        neg_edge_index = sample_zero_entries(data.edge_index, seed)
        train_pos, val_pos, test_pos = data.edge_index, None, None
        train_neg_all, val_neg, test_neg = neg_edge_index, None, None
        train_neg = train_neg_all

    num_train = train_pos.shape[1]

    if method in ['gae', 'dgi']:
        data.x = data.x.to(device)
        train_pos = train_pos.to(device)
        train_neg = train_neg.to(device)

        num_features = data.x.shape[1]
        model = model_class(num_features, dimensions, encoder_class,
                            rec_weight).to(device)

        # Train model
        if ckpt_name is None:
            ckpt_name = 'model'
        print(f'Training {method}')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_auc = 0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = model(data, train_pos, train_neg)
            loss.backward()
            optimizer.step()

            if link_prediction or method == 'gae':
                # Evaluate on val edges
                embeddings = model.encoder(data, train_pos).cpu().detach()
                auc, ap = score_link_prediction(InnerProductScore, embeddings,
                                                val_pos, val_neg, device_str)

                if auc > best_auc:
                    # Keep best model on val set
                    best_auc = auc
                    torch.save(model.state_dict(), ckpt_name)

                if epoch % 50 == 0:
                    log = ('\r[{:03d}/{:03d}] train loss: {:.6f}, '
                           'val_auc: {:6f}, val_ap: {:6f}')
                    print(log.format(epoch, epochs, loss.item(), auc, ap),
                          end='', flush=True)

            elif epoch % 50 == 0:
                log = '\r[{:03d}/{:03d}] train loss: {:.6f}'
                print(log.format(epoch, epochs, loss.item()), end='',
                      flush=True)

            if resample_neg_edges:
                train_neg = sample_edges(train_neg_all,
                                         num_train, seed+epoch).to(device)

        print()

        if not link_prediction and method != 'gae':
            # Save the last state
            torch.save(model.state_dict(), ckpt_name)

        # Evaluate on test edges
        model.load_state_dict(torch.load(ckpt_name))
        os.remove(ckpt_name)
    elif method == 'node2vec':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'node2vec',
                        'data')
        model = Node2Vec(train_pos, path, data.num_nodes)
    elif method == 'graph2gauss':
        model = G2G(data, encoder_str, dimensions[:-1], dimensions[-1],
                    train_pos, val_pos, val_neg, test_pos, test_neg, epochs,
                    lr, K=1, link_prediction=link_prediction)
    else:
        raise ValueError

    if link_prediction:
        if method == 'graph2gauss':
            # graph2gauss link prediction is already evaluated with the KL div
            auc, ap = model.test_auc, model.test_ap
        else:
            model.eval()
            embeddings = model.encoder(data, train_pos).cpu().detach()

            train_pos = train_pos.cpu()
            train_neg = train_neg.cpu()
            if score_class is not InnerProductScore:
                train_val_pos = torch.cat((train_pos, val_pos), dim=-1)
                train_val_neg = torch.cat((train_neg, val_neg), dim=-1)
            else:
                train_val_pos, train_val_neg = None, None

            auc, ap = score_link_prediction(score_class, embeddings,
                                            test_pos, test_neg, device_str,
                                            train_val_pos, train_val_neg,
                                            seed)

        print('test_auc: {:6f}, test_ap: {:6f}'.format(auc, ap))
    else:
        auc, ap = None, None

    return model.encoder, np.array([auc, ap])


ex = Experiment()
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))
else:
    print('Running without Sacred observers')


@ex.config
def config():
    dataset_str = 'cora'
    method = 'dgi'
    encoder_str = 'gcn'
    hidden_dims = [256, 128]
    rec_weight = 0
    lr = 0.001
    epochs = 200
    p_labeled = 0.1
    n_exper = 20
    device = 'cuda'
    timestamp = str(int(time.time()))
    edge_score = 'bilinear'


@ex.capture
def log_statistics(results, metrics, timestamp, _run):
    """Print result statistics and log them with Sacred
    Args:
        - results: numpy array, (n_exper, m) where m is the number of metrics
        - metrics: list of str containing names of the metrics
    """
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    print('-' * 50)
    print('Experiment timestamp: ' + timestamp)
    print('Results')

    for i, m in enumerate(metrics):
        print('{}: {:.1f} ± {:.2f}'.format(m,
                                           mean[i] * 100,
                                           std[i] * 100))
        _run.log_scalar(f'{metrics[i]} mean', mean[i])
        _run.log_scalar(f'{metrics[i]} std', std[i])


@ex.command
def link_pred_experiments(dataset_str, method, encoder_str, hidden_dims,
                          rec_weight, edge_score, lr, epochs, n_exper, device,
                          timestamp, _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 2])
    print('Experiment timestamp: ' + timestamp)
    for i in range(n_exper):
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))
        encoder, scores = train_encoder(dataset_str, method, encoder_str,
                                        hidden_dims, lr, epochs, rec_weight,
                                        device, seed=i, link_prediction=True,
                                        ckpt_name=timestamp,
                                        edge_score=edge_score)
        results[i] = scores

    log_statistics(results, ['AUC', 'AP'])


@ex.automain
def node_class_experiments(dataset_str, method, encoder_str, hidden_dims,
                           rec_weight, lr, epochs, p_labeled, n_exper, device,
                           timestamp, _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 1])
    print('Experiment timestamp: ' + timestamp)
    for i in range(n_exper):
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))
        encoder, _ = train_encoder(dataset_str, method, encoder_str,
                                   hidden_dims, lr, epochs, rec_weight,
                                   device, seed=i, ckpt_name=timestamp)

        data = get_data(dataset_str)
        encoder.to(torch.device('cpu'))
        features = encoder(data, data.edge_index,
                           corrupt=False).detach().numpy()
        labels = data.y.cpu().numpy()
        scores = score_node_classification(features, labels, p_labeled, seed=i)
        test_acc = scores[2]
        print('test_acc: {:.6f}'.format(test_acc))
        results[i] = test_acc

    log_statistics(results, ['ACC'])
