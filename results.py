import os

from pymongo import MongoClient
import matplotlib.pyplot as plt

def get_database():
    """Get a MongoClient using credentials in environment variables """
    user = os.environ.get('MLAB_USR')
    password = os.environ.get('MLAB_PWD')
    database = os.environ.get('MLAB_DB')
    if all([user, password, database]):
        url = f'mongodb://{user}:{password}@ds135812.mlab.com:35812/{database}'
    else:
        raise ConnectionError('Check database environment variables')

    return MongoClient(url)[database]


def get_label_rate_results(database, model_name, id_low, id_high):
    """Get accuracy results for a given model, with different label rates.
    Args:
        - client: a MongoClient to connect with the database
        - model_name: str
        - id_low: int, lowest id (inclusive) to search for results
        - id_high: int, highest id (inclusive) to search for results
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
        if res['config']['model_name'] != model_name:
            continue

        run_id = res['_id']
        train_examples.append(res['config']['train_examples_per_class'])
        acc = next(metrics.find({'run_id': {'$eq': run_id},
                                 'name': {'$eq': 'ACC mean'}}))
        accuracies.append(acc['values'][-1])
        std = next(metrics.find({'run_id': {'$eq': run_id},
                                 'name': {'$eq': 'ACC std'}}))
        stdevs.append(std['values'][-1])

    return train_examples, accuracies, stdevs


def plot_label_rate():
    database = get_database()
    id_low, id_high = 105, 126
    #id_low, id_high = 127, 148

    gae_results = get_label_rate_results(database, 'gae', id_low, id_high)
    dgi_results = get_label_rate_results(database, 'dgi', id_low, id_high)

    plt.plot(gae_results[0], gae_results[1], label='GAE')
    plt.plot(dgi_results[0], dgi_results[1], label='DGI')
    plt.legend()
    plt.show()


plot_label_rate()
