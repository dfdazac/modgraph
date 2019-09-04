import os
import os.path as osp
from collections import OrderedDict, defaultdict

from pymongo import MongoClient
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
from scipy.ndimage.filters import gaussian_filter1d
from incense import ExperimentLoader
import pandas as pd

from modgraph.utils import get_data, adj_from_edge_index


rcParams.update({'font.size': 11})
rc('text', usetex=True)
plt.rc('font', family='serif')

rcParams['axes.axisbelow'] = True

datasets_names = OrderedDict({'amazonphoto': 'Amazon Photo',
                              'amazoncomp': 'Amazon Computer',
                              'coauthorphys': 'Coauthor Physics',
                              'coauthorcs': 'Coauthor CS',
                              'corafull': 'Cora Full',
                              'pubmed': 'Pubmed',
                              'citeseer': 'Citeseer',
                              'cora': 'Cora'})


def get_uri_db_pair():
    uri = os.environ.get('MLAB_URI')
    database = os.environ.get('MLAB_DB')
    if all([uri, database]):
        return uri, database
    else:
        raise ConnectionError('Could not find URI or database')


def get_experiment_loader():
    uri, database = get_uri_db_pair()
    return ExperimentLoader(mongo_uri=uri, db_name=database)


def get_experiment_by_id(exp_id):
    loader = get_experiment_loader()
    ex = loader.find_by_id(exp_id)
    return ex


def get_database():
    """Get a MongoDB database using credentials in environment variables """
    uri, database = get_uri_db_pair()
    return MongoClient(uri)[database]


def get_label_rate_results(database, model_name, timestamp, dataset):
    """Get accuracy results for a given model, with different label rates.

    Args:
        database
        model_name (str): name of the dataset
        timestamp (int): timestamp of the experiments
        dataset (str): name of the dataset to get results for

    Return:
        p_labeled (ndarray): percentage of labeled nodes during training
        accuracies (ndarray)
        stdevs (ndarray)
    """
    runs = database['runs']
    metrics = database["metrics"]

    results = runs.find({'config.timestamp': {'$eq': str(timestamp)}})
    p_labeled = []
    accuracies = []
    stdevs = []
    for res in results:
        if (res['config']['method'] != model_name
                or res['config']['dataset_str'] != dataset
                or res['command'] != 'node_class_experiments'):
            continue

        run_id = res['_id']
        p_labeled.append(res['config']['p_labeled'])
        acc = list(metrics.find({'run_id': {'$eq': run_id},
                                 'name': {'$eq': 'ACC mean'}}))
        accuracies.append(acc[0]['values'][-1])
        std = list(metrics.find({'run_id': {'$eq': run_id},
                                 'name': {'$eq': 'ACC std'}}))
        stdevs.append(std[0]['values'][-1])

    return np.array(p_labeled), np.array(accuracies), np.array(stdevs)


def plot_label_rate(model2ids, dataset, plot_stdevs=False):
    """Make a plot of node classification accuracy vs label rate
    for different models.

    Args:
        model2ids (dict): maps model name (str) to experiments timestamp (int)
        dataset (str): name of the dataset
        plot_stdevs (bool): whether to add regions around 1 standard deviation.
            Default: False
    """
    database = get_database()

    plt.figure(figsize=(2.5, 2.5))
    max_acc = 0
    for model, timestamp in model2ids.items():
        results = get_label_rate_results(database, model.lower(),
                                         timestamp, dataset)
        label = 'G2G' if model == 'graph2gauss' else model
        plt.plot(results[0], results[1], '.-', label=label,
                 linewidth=0.75)
        if np.max(results[1]) > max_acc:
            max_acc = np.max(results[1])
        if plot_stdevs:
            plt.fill_between(results[0], results[1] - results[2],
                             results[1] + results[2], alpha=0.25)

    plt.xticks(results[0], results[0])
    plt.legend(loc='lower right')
    plt.xlabel('Label rate')
    plt.ylabel('Accuracy')
    plt.title(dataset.capitalize())
    plt.tight_layout()
    plt.ylim([0.4, max_acc * 1.05])
    plt.show()


def make_plots():
    model2ids = {'GAE': 1552760104,
                 'DGI': 1552739016,
                 'graph2gauss': 1552822242}
    plot_label_rate(model2ids, 'cora')

    model2ids = {'GAE': 1552760104,
                 'DGI': 1552739016,
                 'graph2gauss': 1552822959}
    plot_label_rate(model2ids, 'citeseer')

    model2ids = {'GAE': 1552760104,
                 'DGI': 1552739016,
                 'graph2gauss': 1552822959}
    plot_label_rate(model2ids, 'pubmed')


def dataset_boxplots():
    X = []
    for dataset in datasets_names:
        path = osp.join('data', dataset)
        data = get_data(dataset, path)
        adj = adj_from_edge_index(data.edge_index, dataset.num_nodes)
        degrees = np.array(adj.sum(axis=0)).squeeze()
        X.append(degrees)

    plt.figure(figsize=(4, 3))
    plt.boxplot(X, showfliers=False, vert=False)
    plt.xlim([0, 92])
    plt.yticks([i+1 for i in range(len(datasets_names))], datasets_names.values())
    plt.xlabel('Node degree')
    plt.tight_layout()
    plt.show(block=False)


def train_save_embeddings(method, dataset_str):
    from train import train

    n_points = 4
    emb_dim = 2
    hidden_dims = [256, emb_dim * n_points]
    embeddings, _ = train(dataset_str, method, encoder_str='mlp',
                          dimensions=hidden_dims, n_points=n_points,
                          lr=1e-3, epochs=2000, link_prediction=False,
                          device_str='cuda', seed=0)
    if method == 'sge':
        embeddings = embeddings.reshape(-1, n_points,
                                        hidden_dims[-1] // n_points)
    embeddings = embeddings.numpy()
    np.save('emb_' + method, embeddings)


def save_cloud_visualization(num_points, embeddings, vis_pos, vis_neg):
    num_samples = embeddings.shape[0]

    emb_dim = embeddings.shape[1] // num_points
    points = embeddings.reshape(num_samples, num_points, -1).numpy()

    means = points.mean(axis=1, keepdims=True)
    distances = np.sqrt(np.sum((points - means) ** 2, axis=-1))
    distances = distances.mean()

    cloud_mean = points.reshape(-1, emb_dim).mean(axis=0, keepdims=True)
    cloud_dists = np.sqrt(np.sum((points.reshape(-1, emb_dim) - cloud_mean)**2,
                                 axis=-1))
    cloud_dists = cloud_dists.mean()

    print(f'Mean cloud variance: {distances:.6f}')
    print(f'Total variance: {cloud_dists:.6f}')

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


def plot_adjacency(method, dataset_str):
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    embeddings = np.load('emb_' + method + '.npy')
    data = get_data(dataset_str)
    adj = adj_from_edge_index(data.edge_index, data.num_nodes)
    fig, ax = plt.subplots()
    ax.imshow(adj.toarray(), cmap='binary')
    fig.savefig('adj')
    plt.cla()

    pred_adj = sigmoid(embeddings @ embeddings.T)
    ax.imshow(pred_adj, cmap='binary')
    fig.savefig('adj_pred')


def sge_curve(dataset_str):
    from train import train

    max_total_dims = 30
    fig, ax = plt.subplots()

    for emb_dim in [0, 2, 3, 4]:
        if emb_dim == 0:
            max_points = max_total_dims
        else:
            max_points = max_total_dims // emb_dim

        n_points = np.arange(1, max_points + 1)

        ap = np.empty(max_points)
        total_dims = np.empty(max_points)

        for i, n in enumerate(n_points):
            print('emb_dim: {:d} points: {:d}'.format(emb_dim, n))
            if emb_dim == 0:
                total_d = n
                points = 1
            else:
                total_d = n * emb_dim
                points = n
            total_dims[i] = total_d
            hidden_dims = [256, total_d]
            _, results = train(dataset_str, method='sge', encoder_str='mlp',
                               dimensions=hidden_dims,
                               n_points=points,
                               lr=1e-2, epochs=200,
                               link_prediction=True,
                               device_str='cuda', seed=0)
            ap[i] = results[1]

        label = '_d' if emb_dim == 0 else str(emb_dim)
        ax.plot(total_dims, ap, label='R{:d}'.format(label))

    plt.legend()
    plt.xlabel('d')
    plt.ylabel('AP')
    fig.savefig('sge_{:d}_1e-2_{}'.format(emb_dim, dataset_str))


def get_graph_assortativity(dataset_str):
    path = osp.join('data', dataset_str)
    data = get_data(dataset_str, path)
    adj = adj_from_edge_index(data.edge_index, data.num_nodes)
    graph = nx.from_scipy_sparse_matrix(adj)

    # Add node attributes
    for i, label in enumerate(data.y):
        graph.nodes[i]['class'] = label.item()

    deg_assort = nx.degree_assortativity_coefficient(graph)
    attr_assor = nx.attribute_assortativity_coefficient(graph, 'class')

    print(f'Graph: {dataset_str}')
    print(f'Degree assortativity: {deg_assort:.6f}')
    print(f'Attribute assortativity: {attr_assor:.6f}')

    return deg_assort, attr_assor


def plot_losses():
    size = (2.5, 2.5)

    # BCE loss
    s = np.linspace(0.01, 0.99, num=100)
    plt.figure(figsize=size)
    plt.plot(s, -np.log(s), label=r'$\mathcal{L}(E)$')
    plt.plot(s, -np.log(1-s), label=r'$\mathcal{L}(\tilde{E})$')
    plt.tight_layout()
    plt.legend(loc='upper right')

    # Square-exponential loss
    s = np.linspace(-2, 2, num=100)
    plt.figure(figsize=size)
    plt.plot(s, s**2, label=r'$\mathcal{L}(E)$')
    plt.plot(s, np.exp(-s), label=r'$\mathcal{L}(\tilde{E})$')
    plt.tight_layout()
    plt.legend(loc='upper right')

    # Hinge loss
    plt.figure(figsize=size)
    loss = 1 + s
    loss[loss < 0] = 0.0
    plt.plot(s, loss,  label=r'$\mathcal{L}(E - \tilde{E})$')
    plt.tight_layout()
    plt.legend(loc='upper right')

    # Square-square loss
    plt.figure(figsize=size)
    neg_loss = 1 - s
    neg_loss[neg_loss < 0] = 0.0
    plt.plot(s, s**2, label=r'$\mathcal{L}(E)$')
    plt.plot(s, neg_loss**2, '--',  label=r'$\mathcal{L}(\tilde{E})$')
    plt.tight_layout()
    plt.legend(loc='upper right')

    plt.show()


plot_losses()


def plot_distortions(ids):
    plt.figure(figsize=(3.2, 2.5))
    for exp_id in ids:
        exp = get_experiment_by_id(exp_id)
        n_points = exp.config['n_points']
        distortion = exp.metrics['loss']
        dist_smooth = gaussian_filter1d(distortion, sigma=4)
        plt.plot(dist_smooth, label='$n = {:d}$'.format(n_points))

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.ylim([0.13, 0.25])
    plt.ylim([0.227, 0.35])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# plot_distortions([273, 274, 275])
# plot_distortions([259, 260, 261])


def plot_assortativity():
    deg_assort_dict = OrderedDict()
    attr_assort_dict = OrderedDict()
    file = open(osp.join('results-data', 'assortativity.csv'))
    # Skip header
    file.readline()
    for line in file:
        dataset_str, deg, attr = line.split(',')
        deg_assort_dict[dataset_str] = float(deg)
        attr_assort_dict[dataset_str] = float(attr)

    deg_assorts = [deg_assort_dict[dataset] for dataset in datasets_names]
    attr_assorts = [attr_assort_dict[dataset] for dataset in datasets_names]

    x = np.array(range(len(datasets_names)))
    h = 0.4
    plt.figure(figsize=(4, 3.5))
    plt.grid()
    plt.rc('axes', axisbelow=True)
    plt.barh(x - h, deg_assorts, height=h, label='Degree', align='edge')
    plt.barh(x, attr_assorts, height=h, label='Attribute', align='edge')
    plt.yticks([i for i in range(len(datasets_names))],
               datasets_names.values())
    plt.xlabel('Assortativity')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
               ncol=2, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()


def plot_train_test_ap(query, max_num_datasets=3):
    num_datasets = len(datasets_names)
    if max_num_datasets < 1 or max_num_datasets > num_datasets:
        raise ValueError(f'max_num_datasets should be between 1'
                         f' and {num_datasets:d}')

    loader = get_experiment_loader()

    experiments = loader.find(query)

    dataset_metrics = defaultdict(dict)

    for exp in experiments:
        dataset = exp.config['dataset_str']
        if dataset not in datasets_names:
            raise ValueError(f'Unrecognized dataset {dataset}')

        encoder = exp.config['encoder_str']
        train_ap = exp.metrics['train AP mean'][0]
        test_ap = exp.metrics['test AP mean'][0]
        dataset_metrics[dataset][encoder] = [train_ap, test_ap]

    metrics = {encoder: {'train': [], 'test': []} for encoder in ['gcn', 'sgc']}
    for dataset in reversed(datasets_names):
        for encoder in metrics:
            train_ap, test_ap = dataset_metrics[dataset][encoder]
            metrics[encoder]['train'].append(train_ap)
            metrics[encoder]['test'].append(test_ap)

    ax_multiplier = 2
    w = 0.3
    x = np.array(range(num_datasets)) * ax_multiplier
    plt.figure()
    plt.bar(x - 2*w, metrics['gcn']['train'], width=0.3, color='C0')
    plt.bar(x - w, metrics['gcn']['test'], width=0.3, color='C1')
    plt.bar(x + w, metrics['sgc']['train'], width=0.3, color='C0')
    plt.bar(x + 2*w, metrics['sgc']['test'], width=0.3, color='C1')
    plt.show()


# conditions = {'$and': [{'config.timestamp': '1558092869'},
#                        {'command': 'link_pred_experiments'},
#                        {'config.method': 'dgi'},
#                        {'config.encoder_str': {'$in': ['gcn', 'sgc']}}
#                        ]
#               }
# plot_train_test_ap(conditions)


def parallel_coordinates_plot(log_id, metrics, xtick_labels, ylabel):
    if len(metrics) != len(xtick_labels):
        raise ValueError('metrics and labels should have the same length.')

    path = osp.join('logs', log_id, 'results.csv')
    df = pd.read_csv(path)

    print('Read data with the following columns:')
    print(df.columns)

    inner_score_idx = df.repr_str.isin(['euclidean_inner', 'euclidean_bilinear'])
    df_product = df[inner_score_idx]
    df_distance = df[~inner_score_idx]
    labels = ['IP', 'Dist']

    plt.figure(figsize=(4, 3.5))
    for i, df in enumerate((df_product, df_distance)):
        df = df[metrics]
        plt.plot(df.transpose(), f'C{i:d}', alpha=0.1, label=labels[i])

    plt.xticks(range(len(metrics)), xtick_labels)
    plt.ylabel(ylabel)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    leg = plt.legend(handles, labels, loc='upper center',
                     bbox_to_anchor=(0.5, 1.20),
                     ncol=2, fancybox=True, shadow=True)
    for lh in leg.legendHandles:
        lh._alpha = 1

    plt.tight_layout()
    plt.show()


# Cora
# parallel_coordinates_plot('1558263980', ['train_ap', 'val_ap'],
#                           ['Train', 'Valid'], 'AP')
# parallel_coordinates_plot('1558295765', ['train_acc', 'Objective'],
#                           ['Train', 'Test'], 'Accuracy')
# # Citeseer
# parallel_coordinates_plot('1558824673', ['train_ap', 'val_ap'],
#                           ['Train', 'Valid'], 'AP')
# parallel_coordinates_plot('1558899609', ['train_acc', 'Objective'],
#                           ['Train', 'Test'], 'Accuracy')


def plot_lsgae_pca(filename):
    w = 0.3
    with open(filename) as file:
        file.readline()
        methods = {}
        for i, line in enumerate(file):
            values = line.split(',')
            method = values[0]
            means = list(map(float, values[1::2]))
            stdevs = list(map(float, values[2::2]))
            methods[method] = (means, stdevs)

            x = np.array(range(len(means)))
            plt.bar(x + i * w, means, width=w)

    plt.show()


# plot_lsgae_pca('results-data/lsgae-pca-lp.csv')


def parallel_coordinates_plot_wasserstein(log_id, encoder, metrics,
                                          xtick_labels, ylabel):
    if len(metrics) != len(xtick_labels):
        raise ValueError('metrics and labels should have the same length.')

    path = osp.join('logs', log_id, 'results.csv')
    df = pd.read_csv(path)

    print('Read data with the following columns:')
    print(df.columns)

    df_idx = df.encoder_str == encoder

    plt.figure(figsize=(2.5, 2.0))
    df = df[df_idx][metrics]
    plt.plot(df.transpose(), 'C0', alpha=0.1)
    max_ap = max(df['test_ap'])
    plt.plot([0, 1], [max_ap, max_ap], 'k--')

    plt.xticks(range(len(metrics)), xtick_labels)
    plt.ylabel(ylabel)
    plt.ylim([0.7, 1.05])
    plt.tight_layout()
    plt.show()


# Cora
# parallel_coordinates_plot_wasserstein('1559740076', 'mlp',
#                                       ['train_ap', 'test_ap'],
#                                       ['Train', 'Test'], 'AP')
# parallel_coordinates_plot_wasserstein('1559740076', 'gcn',
#                                       ['train_ap', 'test_ap'],
#                                       ['Train', 'Test'], 'AP')
# parallel_coordinates_plot_wasserstein('1559740076', 'sgc',
#                                       ['train_ap', 'test_ap'],
#                                       ['Train', 'Test'], 'AP')

# for dataset_str in ['cora', 'citeseer', 'pubmed', 'corafull',
#                     'coauthorcs', 'coauthorphys', 'amazoncomp',
#                     'amazonphoto']:
#     get_graph_assortativity(dataset_str)

# dataset_boxplots()

# plot_assortativity()

# get_graph_assortativity('citeseer')
