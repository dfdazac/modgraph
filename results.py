import os
from collections import OrderedDict

from pymongo import MongoClient
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx

from train import train
from modgraph.utils import get_data, adj_from_edge_index

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica Neue']


def get_database():
    """Get a MongoDB database using credentials in environment variables """
    uri = os.environ.get('MLAB_URI')
    database = os.environ.get('MLAB_DB')
    if all([uri, database]):
        return MongoClient(uri)[database]
    else:
        raise ConnectionError('Check database environment variables')


def get_label_rate_results(database, model_name, timestamp, dataset):
    """Get accuracy results for a given model, with different label rates.

    Args:
        client (MongoClient): to connect with the database
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
    datasets = OrderedDict({'amazonphoto': 'Amazon Photo',
                            'amazoncomp': 'Amazon Computer',
                            'coauthorphys': 'Coauthor Physics',
                            'coauthorcs': 'Coauthor CS',
                            'corafull': 'Cora Full',
                            'pubmed': 'Pubmed',
                            'citeseer': 'Citeseer',
                            'cora': 'Cora'})
    X = []
    for dataset in datasets:
        data = get_data(dataset)
        adj = adj_from_edge_index(data.edge_index)
        degrees = np.array(adj.sum(axis=0)).squeeze()
        X.append(degrees)

    plt.figure(figsize=(4, 3))
    plt.boxplot(X, showfliers=False, vert=False)
    plt.xlim([0, 95])
    plt.yticks([i+1 for i in range(len(datasets))], datasets.values())
    plt.xlabel('Node degree')
    plt.tight_layout()
    plt.show()


def train_save_embeddings(method, dataset_str):
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


def plot_embeddings(method, dataset_str):
    embeddings = np.load('emb_' + method + '.npy')
    data = get_data(dataset_str)
    labels = data.y.numpy()
    n_labels = np.unique(labels).size
    graph = nx.from_scipy_sparse_matrix(adj_from_edge_index(data.edge_index))
    y = data.y.numpy()

    z = TSNE(n_components=2).fit_transform(embeddings)
    pos = {i: z_i for i, z_i in enumerate(z)}
    fig, ax = plt.subplots()
    nx.draw(graph, pos, node_size=50, node_color=y, ax=ax,
            cmap=plt.cm.get_cmap('jet', n_labels), edge_color='silver')
    ax.set_axis_on()
    fig.savefig('fig_' + method)
    plt.cla()


def plot_sge_embeddings(dataset_str):
    embs = np.load('emb_sge.npy')
    n_nodes, n_points, emb_dim = embs.shape
    data = get_data(dataset_str)
    labels = data.y.numpy()
    n_labels = np.unique(labels).size
    y = np.expand_dims(data.y.numpy(), axis=1)

    embs = embs.reshape(n_nodes * n_points, emb_dim)
    labels = np.tile(np.ones_like(y) * y, [1, n_points])
    labels = labels.reshape(n_nodes * n_points)

    fig, ax = plt.subplots()
    ax.scatter(embs[:, 0], embs[:, 1], c=labels,
               cmap=plt.cm.get_cmap('jet', n_labels))

    ax.set_axis_on()
    fig.savefig('fig_sge')
    plt.cla()


def plot_adjacency(method, dataset_str):
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    embeddings = np.load('emb_' + method + '.npy')
    data = get_data(dataset_str)
    adj = adj_from_edge_index(data.edge_index)
    fig, ax = plt.subplots()
    ax.imshow(adj.toarray(), cmap='binary')
    fig.savefig('adj')
    plt.cla()

    pred_adj = sigmoid(embeddings @ embeddings.T)
    ax.imshow(pred_adj, cmap='binary')
    fig.savefig('adj_pred')


def sge_curve(dataset_str):
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


train_save_embeddings('sge', 'cora')
plot_sge_embeddings('cora')

