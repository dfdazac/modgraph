import os

from pymongo import MongoClient
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from train import get_data, train_encoder

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


def get_label_rate_results(database, model_name, id_low, id_high, dataset):
    """Get accuracy results for a given model, with different label rates.
    Args:
        - client: a MongoClient to connect with the database
        - model_name: str
        - id_low: int, lowest id (inclusive) to search for results
        - id_high: int, highest id (inclusive) to search for results
        - dataset: str, name of the dataset to get results for
    Return:
        - train_examples: list, number of labeled nodes during training
        - accuracies: list
        - stdevs: list
    """
    runs = database['runs']
    metrics = database["metrics"]

    # Find runs with id_low <= _id <= id_high
    results = runs.find({'_id': {'$gte': id_low, '$lte': id_high}})
    train_examples = []
    accuracies = []
    stdevs = []
    for res in results:
        if (res['config']['model_name'] != model_name
                or res['config']['dataset'] != dataset):
            continue

        run_id = res['_id']
        train_examples.append(res['config']['train_examples_per_class'])
        acc = next(metrics.find({'run_id': {'$eq': run_id},
                                 'name': {'$eq': 'ACC mean'}}))
        accuracies.append(acc['values'][-1])
        std = next(metrics.find({'run_id': {'$eq': run_id},
                                 'name': {'$eq': 'ACC std'}}))
        stdevs.append(std['values'][-1])

    return np.array(train_examples), np.array(accuracies), np.array(stdevs)


def plot_label_rate(model2ids, dataset):
    database = get_database()

    plt.figure(figsize=(2.5, 2.5))
    for model in model2ids:
        id_low, id_high = model2ids[model]
        results = get_label_rate_results(database, model.lower(),
                                         id_low, id_high, dataset)
        plt.plot(results[0], results[1], '.-', label=model,
                 linewidth=0.75)
        # plt.fill_between(results[0], results[1] - results[2],
        #                 results[1] + results[2], alpha=0.25)

    xticks = [i for i in range(2, np.max(results[0]) + 1, 4)]
    plt.xticks(xticks, xticks)
    plt.legend(loc='lower right')
    plt.xlabel('Nodes per class')
    plt.ylabel('Accuracy')
    plt.title(dataset.capitalize())
    plt.tight_layout()
    plt.ylim([0.4, 0.85])
    plt.show()

model2ids = {'GAE': 1552760104,
             'DGI': 1552739016,
             'G2G': 1552822242}
plot_label_rate(model2ids, 'cora')


def plot_embeddings(model_name, dataset_str):
    _, embeddings = train_encoder(model_name, 'cuda', dataset_str, [256, 128],
                               lr=0.001, epochs=2, random_splits=False,
                               rec_weight=0, encoder='gcn', seed=42,
                               train_examples_per_class=20,
                               val_examples_per_class=30)

    data = get_data(dataset_str, train_examples_per_class=20,
                          val_examples_per_class=30)
    embeddings = embeddings.numpy()

    z = TSNE(n_components=2).fit_transform(embeddings)
    labels = data.y.numpy()
    n_labels = np.unique(labels).size
    plt.scatter(z[:, 0], z[:, 1], c=data.y.numpy(),
                cmap=plt.cm.get_cmap("jet", n_labels))
    plt.savefig('sdf', format='pdf')


#plot_embeddings('dgi', 'cora')


