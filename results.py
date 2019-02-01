import os

from pymongo import MongoClient
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica Neue']
import matplotlib.pyplot as plt
import numpy as np

def get_database():
    """Get a MongoDB database using credentials in environment variables """
    user = os.environ.get('MLAB_USR')
    password = os.environ.get('MLAB_PWD')
    database = os.environ.get('MLAB_DB')
    if all([user, password, database]):
        url = f'mongodb://{user}:{password}@ds135812.mlab.com:35812/{database}'
    else:
        raise ConnectionError('Check database environment variables')

    return MongoClient(url)[database]


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


def plot_label_rate(id_low, id_high, dataset):
    database = get_database()

    gae_results = get_label_rate_results(database, 'gae', id_low, id_high,
                                         dataset)
    dgi_results = get_label_rate_results(database, 'dgi', id_low, id_high,
                                         dataset)

    plt.figure(figsize=(2.5, 2.5))
    plt.plot(gae_results[0], gae_results[1], '.-', label='GAE', linewidth=1)
    plt.fill_between(gae_results[0], gae_results[1] - gae_results[2],
                     gae_results[1] + gae_results[2], alpha=0.25)

    plt.plot(dgi_results[0], dgi_results[1], 'r.--', label='DGI', linewidth=1)
    plt.fill_between(dgi_results[0], dgi_results[1] - dgi_results[2],
                     dgi_results[1] + dgi_results[2], alpha=0.2)

    xticks = [i for i in range(2, np.max(gae_results[0]) + 1, 4)]
    plt.xticks(xticks, xticks)
    plt.legend(loc='lower right')
    plt.xlabel('Nodes per class')
    plt.ylabel('Accuracy')
    plt.title(dataset.capitalize())
    plt.tight_layout()
    plt.show()


plot_label_rate(105, 170, 'cora')
plot_label_rate(105, 170, 'citeseer')
plot_label_rate(105, 170, 'pubmed')
