import os

from pymongo import MongoClient
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

#from train import get_data, train_encoder

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
        - client: a MongoClient to connect with the database
        - model_name: str
        - id_low: int, lowest id (inclusive) to search for results
        - id_high: int, highest id (inclusive) to search for results
        - dataset: str, name of the dataset to get results for
    Return:
        - p_labeled: ndarray, percentage of labeled nodes during training
        - accuracies: ndarray
        - stdevs: ndarray
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


# def plot_embeddings(model_name, dataset_str):
#     _, embeddings = train_encoder(model_name, 'cuda', dataset_str, [256, 128],
#                                lr=0.001, epochs=2, random_splits=False,
#                                rec_weight=0, encoder='gcn', seed=42,
#                                train_examples_per_class=20,
#                                val_examples_per_class=30)
#
#     data = get_data(dataset_str, train_examples_per_class=20,
#                           val_examples_per_class=30)
#     embeddings = embeddings.numpy()
#
#     z = TSNE(n_components=2).fit_transform(embeddings)
#     labels = data.y.numpy()
#     n_labels = np.unique(labels).size
#     plt.scatter(z[:, 0], z[:, 1], c=data.y.numpy(),
#                 cmap=plt.cm.get_cmap("jet", n_labels))
#     plt.savefig('sdf', format='pdf')


#plot_embeddings('dgi', 'cora')


