import os
from datetime import datetime

import torch
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

from utils import (get_data, get_data_splits,
                   score_node_classification,
                   score_link_prediction, link_prediction_scores,
                   score_node_classification_sets)
import models
from samplers import (make_sample_iterator, FirstNeighborSampling,
                      GraphCorruptionSampling, RankedSampling,
                      ShortestPathSampling)


def train_encoder(dataset_str, method, encoder_str, dimensions, n_points, lr,
                  epochs, device_str, link_prediction=False, seed=0,
                  ckpt_name=None, edge_score='inner'):
    if encoder_str == 'mlp':
        encoder_class = models.MLPEncoder
    elif encoder_str == 'gcn':
        encoder_class = models.GCNEncoder
    elif encoder_str == 'sgc':
        encoder_class = models.SGCEncoder
    elif encoder_str == 'mlpgauss':
        encoder_class = models.MLPGaussianEncoder
    elif encoder_str == 'gcngauss':
        encoder_class = models.GCNMLPGaussianEncoder
    else:
        raise ValueError(f'Unknown encoder {encoder_str}')

    energy = 'kldiv'
    if method == 'dgi':
        model_class = models.DGI
    elif method == 'gae':
        model_class = models.GAE
    elif method == 'sge':
        model_class = models.SGEMetric
    elif method == 'graph2gauss':
        model_class = models.G2G
    elif method == 'graph2vec':
        energy = 'sqeuclidean'
        model_class = models.G2G
    elif method in ['node2vec', 'raw']:
        model_class = None
    else:
        raise ValueError(f'Unknown method {method}')

    if edge_score == 'inner':
        edge_score_class = None
    elif edge_score == 'bilinear':
        edge_score_class = models.BilinearScore
    else:
        raise ValueError(f'Unknown edge score {edge_score}')

    if not torch.cuda.is_available() and device_str.startswith('cuda'):
        raise ValueError(f'Device {device_str} specified '
                         'but CUDA is not available')

    device = torch.device(device_str)

    # Load and split data
    if not link_prediction and method == 'gae':
        neg_sample_mult = 10
        resample_neg_edges = True
        # GAE objective is link prediction
        link_prediction = True
    else:
        neg_sample_mult = 1
        resample_neg_edges = False

    add_self_connections = method == 'node2vec'
    data, pos_split, neg_split = get_data_splits(dataset_str, neg_sample_mult,
                                                 link_prediction,
                                                 add_self_connections, seed)
    train_pos, val_pos, test_pos = pos_split
    train_neg, val_neg, test_neg = neg_split

    if method == 'gae':
        train_sampler = FirstNeighborSampling(epochs, train_pos, train_neg, resample_neg_edges)
    elif method == 'dgi':
        train_sampler = GraphCorruptionSampling(epochs, train_pos, data.num_nodes)
    elif method in ['graph2gauss', 'graph2vec']:
        train_sampler = RankedSampling(epochs, train_pos)
    elif method == 'sge':
        train_sampler = ShortestPathSampling(epochs, train_pos)
    elif method == 'node2vec':
        pass
    else:
        raise ValueError(f'Sampling strategy undefined for method {method}')

    x = data.x.to(device)
    edge_index = train_pos.to(device)
    # Train model
    if method in ['gae', 'dgi', 'sge', 'graph2gauss', 'graph2vec']:
        train_iter = make_sample_iterator(train_sampler, num_workers=2)

        num_features = data.x.shape[1]
        encoder = encoder_class(num_features, dimensions)
        emb_dim = dimensions[-1]
        model = model_class(encoder, emb_dim=emb_dim,
                            energy=energy, n_points=n_points).to(device)

        # Train model
        if ckpt_name is None:
            ckpt_name = 'model'
        print(f'Training {method}')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_auc = 0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            pos_samples, neg_samples = next(train_iter)
            pos_samples = pos_samples.to(device)
            neg_samples = neg_samples.to(device)

            loss = model(x, edge_index, pos_samples, neg_samples)
            loss.backward()
            optimizer.step()

            if link_prediction or isinstance(train_sampler, FirstNeighborSampling):
                # Evaluate on val edges
                embeddings = model.encoder(x, edge_index).detach().cpu()
                pos_scores = model.score_pairs(embeddings, val_pos[0], val_pos[1])
                neg_scores = model.score_pairs(embeddings, val_neg[0], val_neg[1])

                auc, ap = link_prediction_scores(pos_scores, neg_scores)

                if auc > best_auc:
                    # Keep best model on val set
                    best_auc = auc
                    torch.save(model.state_dict(), ckpt_name)

                if epoch % 20 == 0:
                    time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    log = ('[{}] [{:03d}/{:03d}] train loss: {:.6f}, '
                           'val_auc: {:6f}, val_ap: {:6f}')
                    print(log.format(time, epoch, epochs, loss.item(),
                                     auc, ap))

            elif epoch % 20 == 0:
                time = datetime.now().strftime("%Y-%m-%d %H:%M")
                log = '[{}] [{:03d}/{:03d}] train loss: {:.6f}'
                print(log.format(time, epoch, epochs, loss.item()))

        if not link_prediction and method != 'gae':
            # Save the last state
            torch.save(model.state_dict(), ckpt_name)

        # Load best checkpoint
        model.load_state_dict(torch.load(ckpt_name))
        os.remove(ckpt_name)
    elif method == 'node2vec':
        model = models.Node2Vec(train_pos, data.num_nodes, dimensions[-1])
    else:
        model = None

    if method == 'raw':
        embeddings = x.cpu()
    else:
        model.eval()
        embeddings = model.encoder(x, edge_index).detach().cpu()

    if link_prediction:
        if edge_score_class is None:
            # Use method score for link prediction
            pos_scores = model.score_pairs(embeddings, test_pos[0], test_pos[1])
            neg_scores = model.score_pairs(embeddings, test_neg[0], test_neg[1])
            auc, ap = link_prediction_scores(pos_scores, neg_scores)
        else:
            if method == 'graph2gauss':
                embeddings = embeddings[0]

            # Train a custom link predictor
            train_pos = train_pos.cpu()
            train_neg = train_neg.cpu()
            train_val_pos = torch.cat((train_pos, val_pos), dim=-1)
            train_val_neg = torch.cat((train_neg, val_neg), dim=-1)

            auc, ap = score_link_prediction(edge_score_class, embeddings,
                                            test_pos, test_neg, device_str,
                                            train_val_pos, train_val_neg,
                                            seed)

        print('test_auc: {:6f}, test_ap: {:6f}'.format(auc, ap))
    else:
        auc, ap = None, None

    if method in ['graph2gauss', 'graph2vec']:
        # Use the mean for downstream tasks
        embeddings = embeddings[0]

    return embeddings, np.array([auc, ap])


ex = Experiment()
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))
else:
    print('Running without Sacred observers')


# noinspection PyUnusedLocal
@ex.config
def config():
    """
    dataset_str (str): one of {'cora', 'citeseer', 'pubmed', 'corafull',
                               'coauthorcs, 'coauthorphys', 'amazoncomp',
                               'amazonphoto'}
    method (str): one of {'gae', 'dgi', 'graph2gauss', 'node2vec'}
    encoder_str (str): one of {'mlp', 'gcn', 'sgc'}
    hidden_dims (list): List with number of units in each layer of the encoder
    n_points (int): Number of support points, only used in Sinkhorn embeddings
    lr (float): learning rate
    epochs (int): number of epochs for training
    p_labeled (float): percentage of labeled nodes used for node classification
    n_exper (int): number of experiments to repeat with different random seeds
    device (str): one of {'cpu', 'cuda'}
    timestamp (int): unique identifier for a set of experiments
    edge_score (str): scoring function used for link prediction. One of
        {'inner', 'bilinear'}
    """
    dataset_str = 'cora'
    method = 'gae'
    encoder_str = 'gcn'
    hidden_dims = [256, 128]
    n_points = 10
    lr = 0.001
    epochs = 200
    p_labeled = 0.1
    n_exper = 20
    device = 'cuda'
    timestamp = str(int(datetime.now().timestamp()))
    edge_score = 'inner'


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
        print('{}: {:.1f} ± {:.2f}'.format(m, mean[i] * 100, std[i] * 100))
        _run.log_scalar(f'{metrics[i]} mean', mean[i])
        _run.log_scalar(f'{metrics[i]} std', std[i])


@ex.command
def link_pred_experiments(dataset_str, method, encoder_str, hidden_dims,
                          n_points, edge_score, lr, epochs, n_exper, device,
                          timestamp, _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 2])
    print('Experiment timestamp: ' + timestamp)
    for i in range(n_exper):
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))
        _, scores = train_encoder(dataset_str, method, encoder_str,
                                  hidden_dims, n_points, lr, epochs,
                                  device, seed=i, link_prediction=True,
                                  ckpt_name=timestamp, edge_score=edge_score)
        results[i] = scores

    log_statistics(results, ['AUC', 'AP'])


@ex.automain
def node_class_experiments(dataset_str, method, encoder_str, hidden_dims,
                           n_points, lr, epochs, p_labeled, n_exper, device,
                           timestamp, _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 1])
    print('Experiment timestamp: ' + timestamp)
    for i in range(n_exper):
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))
        embeddings, _ = train_encoder(dataset_str, method, encoder_str,
                                      hidden_dims, n_points, lr, epochs,
                                      device, seed=i, ckpt_name=timestamp)
        if method == 'sge':
            embeddings = embeddings.reshape(-1, n_points,
                                            hidden_dims[-1]//n_points)

        embeddings = embeddings.numpy()
        data = get_data(dataset_str)
        labels = data.y.cpu().numpy()
        if method == 'sge':
            test_acc = score_node_classification_sets(embeddings, labels,
                                                      models.DeepSetClassifier,
                                                      device, p_labeled,
                                                      seed=i)
        else:
            test_acc = score_node_classification(embeddings, labels, p_labeled,
                                                 seed=i)

        print('test_acc: {:.6f}'.format(test_acc))
        results[i] = test_acc

    log_statistics(results, ['ACC'])
