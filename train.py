import os.path as osp
import os

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from gnnbench import GNNBenchmark
from sacred import Experiment
from sacred.observers import MongoObserver

from utils import score_link_prediction, sample_zero_entries, split_edges
from models import MLPEncoder, GraphEncoder, GAE, DGI, Node2Vec, G2G
from g2g.utils import score_node_classification


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
                  rec_weight, device, link_prediction=False, seed=0):
    if not torch.cuda.is_available() and device.startswith('cuda'):
        raise ValueError(f'Device {device} specified '
                         'but CUDA is not available')

    device = torch.device(device)
    data = get_data(dataset_str)
    neg_edge_index = sample_zero_entries(data.edge_index, seed)

    if link_prediction or method == 'gae':
        add_self_connections = method == 'node2vec'
        train_pos, val_pos, test_pos = split_edges(data.edge_index, seed,
                                                   add_self_connections)
        train_neg, val_neg, test_neg = split_edges(neg_edge_index, seed)
    else:
        train_pos, val_pos, test_pos = data.edge_index, None, None
        train_neg, val_neg, test_neg = neg_edge_index, None, None

    # Create model
    if method == 'dgi':
        model_class = DGI
    elif method == 'gae':
        model_class = GAE
    elif method in ['node2vec', 'graph2gauss']:
        model_class = None
    else:
        raise ValueError(f'Unknown model {method}')

    if encoder_str == 'mlp':
        encoder_class = MLPEncoder
    elif encoder_str == 'gcn':
        encoder_class = GraphEncoder
    else:
        raise ValueError(f'Unknown encoder {encoder_str}')

    if method in ['gae', 'dgi']:
        data.x = data.x.to(device)
        train_pos = train_pos.to(device)
        train_neg = train_neg.to(device)

        num_features = data.x.shape[1]
        model = model_class(num_features, dimensions, encoder_class,
                            rec_weight).to(device)

        # Train model
        ckpt_name = 'model.ckpt'
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
                auc, ap = score_link_prediction(embeddings, val_pos, val_neg)

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
            auc, ap = score_link_prediction(embeddings, test_pos, test_neg)

        print('test_auc: {:6f}, test_ap: {:6f}'.format(auc, ap))
    else:
        auc, ap = None, None

    return model.encoder, np.array([auc, ap])


ex = Experiment()
# Set up database logs
user = os.environ.get('MLAB_USR')
password = os.environ.get('MLAB_PWD')
database = os.environ.get('MLAB_DB')
if all([user, password, database]):
    url = f'mongodb://{user}:{password}@ds135812.mlab.com:35812/{database}'
    ex.observers.append(MongoObserver.create(url, database))
else:
    print('Running without Sacred observers')


@ex.config
def config():
    dataset_str = 'cora'
    method = 'graph2gauss'
    encoder_str = 'mlp'
    hidden_dims = [256, 128]
    rec_weight = 0
    lr = 0.001
    epochs = 200
    p_labeled = 0.1
    n_exper = 20
    device = 'cuda'


@ex.capture
def log_statistics(results, metrics, _run):
    """Print result statistics and log them with Sacred
    Args:
        - results: numpy array, (n_exper, m) where m is the number of metrics
        - metrics: list of str containing names of the metrics
    """
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    print('-' * 50)
    print('Results')

    for i, m in enumerate(metrics):
        print('{}: {:.1f} ± {:.2f}'.format(m,
                                           mean[i] * 100,
                                           std[i] * 100))
        _run.log_scalar(f'{metrics[i]} mean', mean[i])
        _run.log_scalar(f'{metrics[i]} std', std[i])


@ex.command
def link_pred_experiments(dataset_str, method, encoder_str, hidden_dims,
                          rec_weight, lr, epochs, n_exper, device, _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 2])

    for i in range(n_exper):
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))
        encoder, scores = train_encoder(dataset_str, method, encoder_str,
                                        hidden_dims, lr, epochs, rec_weight,
                                        device, seed=i, link_prediction=True)
        results[i] = scores

    log_statistics(results, ['AUC', 'AP'])


@ex.automain
def node_class_experiments(dataset_str, method, encoder_str, hidden_dims,
                          rec_weight, lr, epochs, p_labeled, n_exper, device,
                          _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 1])

    for i in range(n_exper):
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))
        encoder, _ = train_encoder(dataset_str, method, encoder_str,
                                   hidden_dims, lr, epochs, rec_weight,
                                   device, seed=i)

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
