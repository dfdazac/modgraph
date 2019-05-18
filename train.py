import os
from datetime import datetime

from sacred import Experiment
from sacred.observers import MongoObserver

from utils import (get_data, get_data_splits,
                   score_node_classification,
                   train_link_prediction, link_prediction_scores,
                   score_node_classification_sets)

import models
from encoder import *
from representation import *
from loss import *
from sampling import *


def build_method(encoder_str, num_features, dimensions, repr_str, loss_str,
                 sampling_str):
    emb_dim = dimensions[-1]
    if repr_str == 'gaussian':
        emb_dim = dimensions[-1] * 2
        dimensions[-1] = emb_dim

    if encoder_str == 'mlp':
        encoder_class = MLPEncoder
    elif encoder_str == 'gcn':
        encoder_class = GCNEncoder
    elif encoder_str == 'sgc':
        encoder_class = SGCEncoder
    elif encoder_str == 'gcnmlp':
        encoder_class = GCNMLPEncoder
    else:
        raise ValueError(f'Unknown encoder {encoder_str}')

    encoder = encoder_class(num_features, dimensions)

    if repr_str == 'euclidean_inner':
        representation = EuclideanInnerProduct()
    elif repr_str == 'euclidean_bilinear':
        representation = EuclideanBilinear(in_features=emb_dim)
    elif repr_str == 'euclidean_distance':
        representation = EuclideanDistance()
    elif repr_str == 'gaussian':
        representation = Gaussian()
    else:
        raise ValueError(f'Unknown representation {repr_str}')

    if loss_str == 'bce_loss':
        loss = bce_loss
    elif loss_str == 'square_exponential':
        loss = square_exponential
    else:
        raise ValueError(f'Unknown loss {loss_str}')

    if sampling_str == 'first_neighbor':
        sampling_class = FirstNeighborSampling
    elif sampling_str == 'graph_corruption':
        sampling_class = GraphCorruptionSampling
    elif sampling_str == 'ranked':
        sampling_class = RankedSampling
    else:
        raise ValueError(f'Unknown sampling {sampling_str}')

    return models.EmbeddingMethod(encoder, representation, loss, sampling_class)


def train(dataset, method, lr, epochs, device_str, link_prediction=False,
          seed=0, ckpt_name=None, edge_score='inner'):
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
    if not link_prediction and method.sampling_class == FirstNeighborSampling:
        neg_sample_mult = 10
        resample_neg = True
        # GAE objective is link prediction
        link_prediction = True
    else:
        neg_sample_mult = 1
        resample_neg = False

    add_self_connections = method == 'node2vec'
    pos_split, neg_split = get_data_splits(dataset, neg_sample_mult,
                                           link_prediction,
                                           add_self_connections, seed)
    train_pos, val_pos, test_pos = pos_split
    train_neg, val_neg, test_neg = neg_split

    train_sampler = method.sampling_class(epochs, train_pos,
                                          neg_index=train_neg,
                                          resample=resample_neg,
                                          num_nodes=dataset.num_nodes)

    x = dataset.x.to(device)
    edge_index = train_pos.to(device)
    # Train model
    if isinstance(method, models.EmbeddingMethod):
        train_iter = make_sample_iterator(train_sampler, num_workers=2)

        if ckpt_name is None:
            ckpt_name = 'model'

        print(f'Training {method}')

        method.to(device)
        optimizer = torch.optim.Adam(method.parameters(), lr=lr)
        best_auc = 0
        for epoch in range(1, epochs + 1):
            method.train()
            optimizer.zero_grad()

            pos_samples, neg_samples = next(train_iter)
            pos_samples = pos_samples.to(device)
            neg_samples = neg_samples.to(device)
            loss = method(x, edge_index, pos_samples, neg_samples)
            loss.backward()
            optimizer.step()

            if link_prediction or method.sampling_class == FirstNeighborSampling:
                # Evaluate on val edges
                with torch.no_grad():
                    pos_scores = method.score_pairs(x, edge_index, val_pos)
                    neg_scores = method.score_pairs(x, edge_index, val_neg)

                auc, ap = link_prediction_scores(pos_scores, neg_scores)

                if auc > best_auc:
                    # Keep best model on val set
                    best_auc = auc
                    torch.save(method.state_dict(), ckpt_name)

                if epoch % 50 == 0:
                    time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    log = ('[{}] [{:03d}/{:03d}] train loss: {:.6f}, '
                           'val_auc: {:6f}, val_ap: {:6f}')
                    print(log.format(time, epoch, epochs, loss.item(),
                                     auc, ap))

            elif epoch % 50 == 0:
                time = datetime.now().strftime("%Y-%m-%d %H:%M")
                log = '[{}] [{:03d}/{:03d}] train loss: {:.6f}'
                print(log.format(time, epoch, epochs, loss.item()))

        if not link_prediction and not isinstance(method.sampling_class, FirstNeighborSampling):
            # Save the last state
            torch.save(method.state_dict(), ckpt_name)

        # Load best checkpoint
        method.load_state_dict(torch.load(ckpt_name))
        os.remove(ckpt_name)
    elif method == 'node2vec':
        # FIXME
        model = None # models.Node2Vec(train_pos, data.num_nodes, dimensions[-1])
    else:
        model = None

    if method == 'raw':
        embeddings = x.cpu()
    else:
        method.eval()
        embeddings = method.encoder(x, edge_index).detach().cpu()

        # if method in ['sge', 'sgemetric']:
        #     points = embeddings.reshape(data.num_nodes, n_points, -1)
        #     mean = points.mean(dim=1, keepdim=True)
        #     stdev = torch.sqrt(torch.sum((points - mean)**2, dim=-1)).mean()
        #     print('Average distance from mean: {:.6f}'.format(stdev.item()))

    results = -np.ones([3, 2])
    if link_prediction:
        if edge_score_class is None:
            # Evaluate on training, validation and test splits
            for i, (pos, neg) in enumerate(zip(pos_split, neg_split)):
                # pos_scores = method.score_pairs(x, edge_index, pos)
                pos_scores = method.representation.score_link_pred(embeddings, pos)
                # neg_scores = method.score_pairs(x, edge_index, neg)
                neg_scores = method.representation.score_link_pred(embeddings, neg)
                results[i] = link_prediction_scores(pos_scores, neg_scores)
        else:
            if method in ['graph2gauss', 'graph2vec']:
                # Use the mean for downstream evaluation
                embeddings = embeddings[0]

            # Train a custom link predictor
            train_pos = train_pos.cpu()
            train_neg = train_neg.cpu()
            train_val_pos = torch.cat((train_pos, val_pos), dim=-1)
            train_val_neg = torch.cat((train_neg, val_neg), dim=-1)

            scores = train_link_prediction(edge_score_class, embeddings,
                                           train_val_pos, train_val_neg,
                                           test_pos, test_neg, device_str,
                                           seed)

            results[0] = scores[:2]
            results[2] = scores[2:]

        for (auc_res, ap_res), split in zip(results, ['train', 'val', 'test']):
            print('{:5} - auc: {:.6f} ap: {:.6f}'.format(split, auc_res, ap_res))

    if method in ['graph2gauss', 'graph2vec']:
        # Use the mean for downstream tasks
        embeddings = embeddings[0]

    return embeddings, results


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

    encoder_str = 'mlp'
    repr_str = 'euclidean_distance'
    loss_str = 'square_exponential'
    sampling_str = 'ranked'

    dimensions = [256, 128]
    n_points = 1
    edge_score = 'inner'
    lr = 0.001
    epochs = 200
    n_exper = 20
    device = 'cuda'
    timestamp = str(int(datetime.now().timestamp()))
    p_labeled = 0.1


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
        print('{:12}: {:.1f} ± {:.2f}'.format(m, mean[i] * 100, std[i] * 100))
        _run.log_scalar(f'{metrics[i]} mean', mean[i])
        _run.log_scalar(f'{metrics[i]} std', std[i])


@ex.command
def link_pred_experiments(dataset_str, encoder_str, dimensions, repr_str,
                          loss_str, sampling_str, n_points, edge_score, lr,
                          epochs, n_exper, device, timestamp, _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 6])
    print('Experiment timestamp: ' + timestamp)
    for i in range(n_exper):
        if i == 16:
            print('wait')
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))

        dataset = get_data(dataset_str)
        method = build_method(encoder_str, dataset.num_features, dimensions,
                              repr_str, loss_str, sampling_str)
        _, scores = train(dataset, method, lr, epochs,
                          device, link_prediction=True, seed=i,
                          ckpt_name=timestamp, edge_score=edge_score)

        results[i] = scores.reshape(-1)

    log_statistics(results, ['train AUC', 'train AP',
                             'valid AUC', 'valid AP',
                             'test AUC', 'test AP'])


@ex.automain
def node_class_experiments(dataset_str, encoder_str, dimensions, repr_str,
                           loss_str, sampling_str, n_points, lr, epochs,
                           p_labeled, n_exper, device, timestamp, _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 2])
    print('Experiment timestamp: ' + timestamp)
    for i in range(n_exper):
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))

        dataset = get_data(dataset_str)
        method = build_method(encoder_str, dataset.num_features, dimensions,
                              repr_str, loss_str, sampling_str)
        embeddings, _ = train(dataset, method, lr, epochs,
                              device, seed=i, ckpt_name=timestamp)
        # if method == 'sge':
        #     embeddings = embeddings.reshape(-1, n_points,
        #                                     hidden_dims[-1]//n_points)

        embeddings = embeddings.numpy()
        data = get_data(dataset_str)
        labels = data.y.cpu().numpy()
        if method == 'sge':
            test_acc = score_node_classification_sets(embeddings, labels,
                                                      models.DeepSetClassifier,
                                                      device, p_labeled,
                                                      seed=i)
        else:
            train_acc, test_acc = score_node_classification(embeddings,
                                                            labels, p_labeled,
                                                            seed=i)

        print('train - acc: {:.6f}'.format(train_acc))
        print('test  - acc: {:.6f}'.format(test_acc))
        results[i] = [train_acc, test_acc]

    log_statistics(results, ['train ACC', 'test ACC'])
