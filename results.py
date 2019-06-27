import os
import os.path as osp
from collections import OrderedDict

from pymongo import MongoClient
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx
from scipy.ndimage.filters import gaussian_filter1d
from incense import ExperimentLoader

from modgraph.utils import get_data, adj_from_edge_index

# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Helvetica Neue']

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


def get_experiment(exp_id):
    uri, database = get_uri_db_pair()
    loader = ExperimentLoader(mongo_uri=uri, db_name=database)
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
        adj = adj_from_edge_index(data.edge_index)
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
    adj = adj_from_edge_index(data.edge_index)
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
    plt.plot(s, -np.log(s), 'k', label=r'$\mathcal{L}(S)$')
    plt.plot(s, -np.log(1-s), 'k--', label=r'$\mathcal{L}(\tilde{S})$')
    plt.tight_layout()
    #plt.legend(loc='upper right')

    # Square-exponential loss
    s = np.linspace(-2, 2, num=100)
    plt.figure(figsize=size)
    plt.plot(s, s**2, 'k',  label=r'$\mathcal{L}(S)$')
    plt.plot(s, np.exp(-s), 'k--',  label=r'$\mathcal{L}(\tilde{S})$')
    plt.tight_layout()
    #plt.legend(loc='upper right')

    # Hinge loss
    plt.figure(figsize=size)
    loss = 1 + s
    loss[loss < 0] = 0.0
    plt.plot(s, loss,  'k', label=r'$\mathcal{L}(S - \tilde{S})$')
    plt.tight_layout()
    #plt.legend(loc='upper right')

    # Square-square loss
    plt.figure(figsize=size)
    neg_loss = 1 - s
    neg_loss[neg_loss < 0] = 0.0
    plt.plot(s, s**2, label=r'$\mathcal{L}(S)$')
    plt.plot(s, neg_loss**2, '--',  label=r'$\mathcal{L}(\tilde{S})$')
    plt.tight_layout()
    #plt.legend(loc='upper right')

    plt.show()


def plot_distortions(ids):
    plt.figure(figsize=(3.2, 2.5))
    for exp_id in ids:
        exp = get_experiment(exp_id)
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
              ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()


# plot_distortions([273, 274, 275])
# plot_distortions([259, 260, 261])

# for dataset_str in ['cora', 'citeseer', 'pubmed', 'corafull',
#                     'coauthorcs', 'coauthorphys', 'amazoncomp',
#                     'amazonphoto']:
#     get_graph_assortativity(dataset_str)

dataset_boxplots()

plot_assortativity()
