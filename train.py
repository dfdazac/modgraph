import os
import os.path as osp
from datetime import datetime
import random
import torch
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import modgraph
import modgraph.utils as utils

root = osp.dirname(osp.realpath(__file__))
ex = Experiment()
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))
else:
    print('Running without Sacred observers')


def build_method(encoder_str, num_features, dimensions, n_points, repr_str,
                 loss_str, sampling_str):
    emb_dim = dimensions[-1]
    if repr_str in ['gaussian', 'gaussian_variational', 'gaussian_flow']:
        # Meand and variance
        emb_dim = dimensions[-1] * 2
    elif repr_str == 'spherical_variational':
        # Mean and concentration
        emb_dim = dimensions[-1] + 1
    dimensions = dimensions[:-1] + [emb_dim]

    if encoder_str == 'mlp':
        encoder_class = modgraph.MLPEncoder
    elif encoder_str == 'gcn':
        encoder_class = modgraph.GCNEncoder
    elif encoder_str == 'sgc':
        encoder_class = modgraph.SGCEncoder
    elif encoder_str == 'gcnmlp':
        encoder_class = modgraph.GCNMLPEncoder
    else:
        raise ValueError(f'Unknown encoder {encoder_str}')

    encoder = encoder_class(num_features, dimensions)

    if repr_str == 'euclidean_inner':
        representation = modgraph.EuclideanInnerProduct()
    elif repr_str == 'euclidean_infomax':
        representation = modgraph.EuclideanInfomax(in_features=emb_dim)
    elif repr_str == 'euclidean_distance':
        representation = modgraph.EuclideanDistance()
    elif repr_str == 'gaussian':
        representation = modgraph.Gaussian()
    elif repr_str == 'gaussian_variational':
        representation = modgraph.GaussianVariational()
    elif repr_str == 'spherical_variational':
        representation = modgraph.HypersphericalVariational()
    elif repr_str == 'point_cloud':
        representation = modgraph.PointCloud(n_points)
    elif repr_str == 'gaussian_flow':
        representation = modgraph.GaussianFlow(in_features=emb_dim)
    else:
        raise ValueError(f'Unknown representation {repr_str}')

    loss = getattr(modgraph, loss_str, None)
    if loss is None:
        raise ValueError(f'Unknown loss {loss_str}')

    if sampling_str == 'first_neighbors':
        sampling_class = modgraph.FirstNeighborSampling
    elif sampling_str == 'graph_corruption':
        sampling_class = modgraph.GraphCorruptionSampling
    elif sampling_str == 'ranked':
        sampling_class = modgraph.RankedSampling
    elif sampling_str == 'shortest_path':
        sampling_class = modgraph.ShortestPathSampling
    else:
        raise ValueError(f'Unknown sampling {sampling_str}')

    return modgraph.EmbeddingMethod(encoder, representation, loss, sampling_class)


def save_cloud_visualization(method, embeddings, vis_pos, vis_neg, epoch):
    num_samples = embeddings.shape[0]
    if hasattr(method.representation, 'n_points'):
        num_points = method.representation.n_points
    else:
        num_points = 1

    emb_dim = embeddings.shape[1] // num_points
    points = embeddings.reshape(num_samples, num_points, -1).numpy()

    means = points.mean(axis=1, keepdims=True)
    distances = np.sqrt(np.sum((points - means) ** 2, axis=-1))
    distances = distances.mean()

    cloud_mean = points.reshape(-1, emb_dim).mean(axis=0, keepdims=True)
    cloud_dists = np.sqrt(np.sum((points.reshape(-1, emb_dim) - cloud_mean)**2,
                                 axis=-1))
    cloud_dists = cloud_dists.mean()

    # print(f'Epoch {epoch:d}')
    # print(f'Mean cloud variance: {distances:.6f}')
    # print(f'Total variance: {cloud_dists:.6f}')

    node_idx = np.unique(np.concatenate((vis_pos, vis_neg), axis=1))
    idx_to_pos = {idx: pos for pos, idx in enumerate(node_idx)}
    points = points[node_idx]
    if emb_dim > 2:
        if emb_dim > 64:
            reducer = TSNE(n_components=2, init='pca', random_state=0)
        else:
            reducer = PCA(n_components=2)

        points = reducer.fit_transform(points.reshape(-1, emb_dim))
    else:
        points = points.reshape(-1, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.scatter(points[:, 0], points[:, 1], color='lightgray')
    ax2.scatter(points[:, 0], points[:, 1], color='lightgray')

    points = points.reshape(-1, num_points, 2)
    colors = ['C0', 'C1', 'C2']
    titles = ('Positive pairs', 'Negative pairs')

    for ax, vis, title in zip((ax1, ax2), (vis_pos, vis_neg), titles):
        for i, color in enumerate(colors):
            cloud_a_idx = idx_to_pos[vis[0, i]]
            cloud_a = points[cloud_a_idx]
            ax.scatter(cloud_a[:, 0], cloud_a[:, 1], s=60, color=color,
                       marker='o', edgecolors='k')

            cloud_a_idx = idx_to_pos[vis[1, i]]
            cloud_a = points[cloud_a_idx]
            ax.scatter(cloud_a[:, 0], cloud_a[:, 1], s=60, color=color,
                       marker='X', edgecolors='k')

        ax.set_axis_on()
        ax.set_title(title)

    fig.savefig('cloud')


def save_embedding_visualization(method, embeddings, labels):
    num_samples = embeddings.shape[0]
    if hasattr(method.representation, 'n_points'):
        num_points = method.representation.n_points
    else:
        num_points = 1

    emb_dim = embeddings.shape[1] // num_points
    points = embeddings.reshape(num_samples, num_points, -1).numpy()

    if emb_dim > 2:
        if emb_dim > 64:
            reducer = TSNE(n_components=2, init='pca', random_state=0)
        else:
            reducer = PCA(n_components=2)

        points = reducer.fit_transform(points.reshape(-1, emb_dim))
    else:
        points = points.reshape(-1, 2)

    n_labels = np.unique(labels).size
    plt.scatter(points[:, 0], points[:, 1], c=labels,
                cmap=plt.cm.get_cmap('jet', n_labels))
    plt.savefig('embeddings')


@ex.capture
def train(dataset, method, lr, epochs, device_str, _run, link_prediction=False,
          seed=0, ckpt_name='model', edge_score='inner', plot_loss=False):
    if edge_score == 'inner':
        edge_score_class = None
    elif edge_score == 'bilinear':
        edge_score_class = modgraph.BilinearScore
    else:
        raise ValueError(f'Unknown edge score {edge_score}')

    if not torch.cuda.is_available() and device_str.startswith('cuda'):
        raise ValueError(f'Device {device_str} specified '
                         'but CUDA is not available')

    device = torch.device(device_str)

    # Load and split data
    neg_sample_mult = 1
    resample_neg = False
    if method != 'node2vec':
        if method.sampling_class == modgraph.FirstNeighborSampling:
            neg_sample_mult = 10
            resample_neg = True
            # GAE objective is link prediction
            link_prediction = True

    add_self_connections = method == 'node2vec'
    pos_split, neg_split = utils.get_data_splits(dataset, neg_sample_mult,
                                                 link_prediction,
                                                 add_self_connections, seed)
    train_pos, val_pos, test_pos = pos_split
    train_neg, val_neg, test_neg = neg_split

    x = dataset.x.to(device)
    edge_index = train_pos.to(device)
    # Train model
    if isinstance(method, modgraph.EmbeddingMethod):
        train_sampler = method.sampling_class(epochs, train_pos,
                                              neg_index=train_neg,
                                              resample=resample_neg,
                                              num_nodes=dataset.num_nodes)
        train_iter = modgraph.make_sample_iterator(train_sampler, num_workers=2)

        rand_postfix = str(random.randint(0, 100000))
        ckpt_path = osp.join(root, ckpt_name + rand_postfix + '.pt')

        print(f'Training {method}')

        method.to(device)
        optimizer = torch.optim.Adam(method.parameters(), lr=lr)
        best_auc = 0
        best_epoch = 0

        # Sample positive and negative pairs for visualization
        num_samples = 100
        sample_idx = np.random.choice(range(train_pos.shape[1]), num_samples,
                                      replace=False)
        vis_pos = train_pos[:, sample_idx].numpy()
        vis_neg = train_neg[:, sample_idx].numpy()
        for epoch in range(1, epochs + 1):
            # if epoch == epochs:
            #     with torch.no_grad():
            #         embeddings = method.embed(x, edge_index).detach().cpu()
            #         # save_cloud_visualization(method, embeddings,
            #         #                          vis_pos, vis_neg, epoch)
            #         save_embedding_visualization(method, embeddings,
            #                                      dataset.y.numpy())

            method.train()
            optimizer.zero_grad()

            pos_samples, neg_samples = next(train_iter)
            # Use these to test that we can overfit:
            # pos_samples = torch.tensor(vis_pos, dtype=torch.long)
            # neg_samples = torch.tensor(vis_neg, dtype=torch.long)
            pos_samples = pos_samples.to(device)
            neg_samples = neg_samples.to(device)
            loss = method(x, edge_index, pos_samples, neg_samples)
            loss.backward()
            optimizer.step()

            if plot_loss:
                _run.log_scalar('loss', loss.item(), epoch)

            if link_prediction or method.sampling_class == modgraph.FirstNeighborSampling:
                # Evaluate on val edges
                with torch.no_grad():
                    pos_scores = method.score_pairs(x, edge_index, val_pos)
                    neg_scores = method.score_pairs(x, edge_index, val_neg)

                auc, ap = utils.link_prediction_scores(pos_scores, neg_scores)

                if auc > best_auc:
                    # Keep best model on val set
                    best_auc = auc
                    best_epoch = epoch
                    torch.save(method.state_dict(), ckpt_path)

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

        if not link_prediction and not isinstance(method.sampling_class, modgraph.FirstNeighborSampling):
            # Save the last state
            torch.save(method.state_dict(), ckpt_path)

        # Load best checkpoint
        print(f'Best AUC obtained at epoch {best_epoch}')
        method.load_state_dict(torch.load(ckpt_path))
        os.remove(ckpt_path)
    elif method == 'node2vec':
        method = modgraph.Node2Vec(train_pos, dataset.num_nodes).to(device)
    else:
        raise ValueError('Unknown method')

    if method == 'raw':
        embeddings = x.cpu()
    else:
        method.eval()
        embeddings = method.encoder(x, edge_index)  # .detach().cpu()

    results = -np.ones([3, 2])
    if link_prediction:
        if edge_score_class is None:
            # Evaluate on training, validation and test splits
            labels = ['train', 'val', 'test']
            for i, (pos, neg) in enumerate(zip(pos_split, neg_split)):
                # pos_scores = method.score_pairs(x, edge_index, pos)
                pos_scores = method.representation.score_link_pred(embeddings, pos)
                # neg_scores = method.score_pairs(x, edge_index, neg)
                neg_scores = method.representation.score_link_pred(embeddings, neg)
                results[i] = utils.link_prediction_scores(pos_scores, neg_scores)

                utils.plot_pr_curve(pos_scores, neg_scores, labels[i])
        else:
            if method in ['graph2gauss', 'graph2vec']:
                # Use the mean for downstream evaluation
                embeddings = embeddings[0]

            # Train a custom link predictor
            train_pos = train_pos.cpu()
            train_neg = train_neg.cpu()
            train_val_pos = torch.cat((train_pos, val_pos), dim=-1)
            train_val_neg = torch.cat((train_neg, val_neg), dim=-1)

            scores = utils.train_link_prediction(edge_score_class, embeddings,
                                           train_val_pos, train_val_neg,
                                           test_pos, test_neg, device_str,
                                           seed)

            results[0] = scores[:2]
            results[2] = scores[2:]

        for (auc_res, ap_res), split in zip(results, ['train', 'val', 'test']):
            print('{:5} - auc: {:.6f} ap: {:.6f}'.format(split, auc_res, ap_res))

    if isinstance(method.representation,
                  (modgraph.Gaussian, modgraph.GaussianVariational)):
        # Use the mean for downstream tasks
        emb_dim = embeddings.shape[1] // 2
        mu, logsigma = torch.split(embeddings, emb_dim, dim=1)
        embeddings = mu

    return embeddings, results


# noinspection PyUnusedLocal
@ex.config
def config():
    """
    dataset_str (str): {'cora', 'citeseer', 'pubmed', 'corafull',
                        'coauthorcs, 'coauthorphys', 'amazoncomp',
                        'amazonphoto'}
    encoder_str (str): {'mlp', 'gcn', 'gcnmlp'}
    repr_str (str): {'euclidean_inner', 'euclidean_infomax', 'gaussian',
                     'euclidean_distance'}
    loss_str (str): {'bce_loss', 'square_exponential'}
    sampling_str (str): {'first_neighbors', 'ranked', 'graph_corruption'}
    dimensions (list): List with number of units in each layer of the encoder
    edge_score (str): scoring function used for link prediction. One of
        {'inner', 'bilinear'}
    lr (float): learning rate
    epochs (int): number of epochs for training
    train_node2vec (bool): if True, ignore method and train with node2vec
    p_labeled (float): percentage of labeled nodes used for node classification
    n_exper (int): number of experiments to repeat with different random seeds
    device (str): one of {'cpu', 'cuda'}
    timestamp (str): unique identifier for a set of experiments
    """
    dataset_str = 'cora'

    encoder_str = 'gcn'
    repr_str = 'euclidean_infomax'
    loss_str = 'bce_loss'
    sampling_str = 'first_neighbors'

    dimensions = [256, 128]
    n_points = 4
    edge_score = 'inner'
    classifier = 'set'
    lr = 0.001
    epochs = 200
    train_node2vec = False
    p_labeled = 0.1
    n_exper = 20
    plot_loss = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    timestamp = str(int(datetime.now().timestamp()))


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


@ex.capture
def get_study(timestamp, _run):
    import sherpa

    parameters = [sherpa.Choice('encoder_str',
                                range=['mlp', 'gcn', 'sgc']),
                  sherpa.Choice('loss_str',
                                range=['bce_loss',
                                       'square_exponential_loss',
                                       'square_square_loss',
                                       'hinge_loss']),
                  sherpa.Choice('sampling_str',
                                range=['first_neighbors',
                                       'ranked',
                                       'graph_corruption']),
                  sherpa.Choice('n_points',
                                range=[1, 4, 8])]

    algorithm = sherpa.algorithms.GridSearch()

    # If Sacred is run with no observers, don't log with sherpa either
    if len(_run.observers) > 0:
        output_dir = osp.join('./logs', timestamp)
        if not osp.isdir(output_dir):
            os.mkdir(output_dir)
    else:
        output_dir = None

    study = sherpa.Study(parameters, algorithm, lower_is_better=False,
                         output_dir=output_dir)
    return study


@ex.command
def modular_search_link_pred():
    study = get_study()

    for trial in study:
        results = link_pred_experiments(**trial.parameters)

        # Compute mean across experiments,
        # and reshape with one row per split (train/val/test)
        results = np.mean(results, axis=0).reshape([3, 2])
        # Objective is test_auc + test_ap
        objective = float(np.sum(results[-1]))
        study.add_observation(trial, iteration=0, objective=objective,
                              context={'train_auc': results[0, 0],
                                       'val_auc': results[1, 0],
                                       'test_auc': results[2, 0],
                                       'train_ap': results[0, 1],
                                       'val_ap': results[1, 1],
                                       'test_ap': results[2, 1]})
        study.finalize(trial)

    if study.output_dir is not None:
        study.save()


@ex.command
def modular_search_node_class():
    study = get_study()

    for trial in study:
        results = node_class_experiments(**trial.parameters)

        # Compute mean across experiments
        results = np.mean(results, axis=0)
        # Objective is test_acc
        objective = float(results[1])
        study.add_observation(trial, iteration=0, objective=objective,
                              context={'train_acc': results[0]})
        study.finalize(trial)

    if study.output_dir is not None:
        study.save()


def load_modular_results(timestamp):
    import sherpa

    output_dir = osp.join('./logs', str(timestamp))
    # noinspection PyUnusedLocal
    study = sherpa.Study.load_dashboard(output_dir)

    wait = ''
    while wait != 'q':
        wait = input('Enter q to quit: ')


load_modular_results = ex.command(load_modular_results, unobserved=True)


@ex.command
def parallel_trial(dataset_str, edge_score, lr, epochs, n_exper, device,
                   timestamp, _run):
    import sherpa

    client = sherpa.Client()
    trial = client.get_trial()

    results = link_pred_experiments(dataset_str, **trial.parameters,
                                    edge_score=edge_score, lr=lr,
                                    epochs=epochs, train_node2vec=False,
                                    n_exper=n_exper, device=device,
                                    timestamp=timestamp)

    # Compute mean across experiments,
    # and reshape with one row per split (train/val/test)
    results = np.mean(results, axis=0).reshape([3, 2])
    # Objective is test_auc + test_ap
    objective = float(np.sum(results[-1]))
    client.send_metrics(trial, iteration=0, objective=objective,
                        context={'train_auc': results[0, 0],
                                 'val_auc': results[1, 0],
                                 'test_auc': results[2, 0],
                                 'train_ap': results[0, 1],
                                 'val_ap': results[1, 1],
                                 'test_ap': results[2, 1]})


@ex.command
def link_pred_experiments(dataset_str, encoder_str, dimensions, n_points, repr_str,
                          loss_str, sampling_str, edge_score, lr,
                          epochs, train_node2vec, n_exper, device, timestamp,
                          _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 6])
    print('Experiment timestamp: ' + timestamp)

    path = osp.join(root, 'data', dataset_str)
    dataset = utils.get_data(dataset_str, path)

    for i in range(n_exper):
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))

        if train_node2vec:
            method = 'node2vec'
        else:
            method = build_method(encoder_str, dataset.num_features, dimensions,
                                  n_points, repr_str, loss_str, sampling_str)
        _, scores = train(dataset, method, lr, epochs,
                          device, link_prediction=True, seed=i,
                          ckpt_name=timestamp, edge_score=edge_score)

        results[i] = scores.reshape(-1)

    log_statistics(results, ['train AUC', 'train AP',
                             'valid AUC', 'valid AP',
                             'test AUC', 'test AP'])

    return results


@ex.command
def node_class_experiments(dataset_str, encoder_str, repr_str, loss_str, sampling_str,
                           dimensions, n_points, classifier, lr, epochs, train_node2vec,
                           p_labeled, n_exper, device, timestamp, _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    results = np.empty([n_exper, 2])
    print('Experiment timestamp: ' + timestamp)

    path = osp.join(root, 'data', dataset_str)
    dataset = utils.get_data(dataset_str, path)

    for i in range(n_exper):
        print('\nTrial {:d}/{:d}'.format(i + 1, n_exper))

        if train_node2vec:
            method = 'node2vec'
        else:
            method = build_method(encoder_str, dataset.num_features, dimensions,
                                  n_points, repr_str, loss_str, sampling_str)
        embeddings, _ = train(dataset, method, lr, epochs,
                              device, seed=i, ckpt_name=timestamp)

        embeddings = embeddings.numpy()
        labels = dataset.y.cpu().numpy()

        if isinstance(method.representation, modgraph.PointCloud) \
                and classifier == 'set':
            embeddings = embeddings.reshape(dataset.num_nodes, n_points, -1)

            scores = utils.score_node_classification_sets(embeddings, labels,
                                                          modgraph.DeepSetClassifier,
                                                          device, p_labeled,
                                                          seed=i)
        else:
            scores = utils.score_node_classification(embeddings, labels,
                                                     p_labeled, seed=i)

        train_acc, test_acc = scores

        print('train - acc: {:.6f}'.format(train_acc))
        print('test  - acc: {:.6f}'.format(test_acc))
        results[i] = [train_acc, test_acc]

    log_statistics(results, ['train ACC', 'test ACC'])

    return results


@ex.automain
def stub():
    return
