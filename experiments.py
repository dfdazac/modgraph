from argparse import ArgumentParser
from datetime import datetime

import torch
from train import ex
from sklearn.model_selection import ParameterGrid

import pprint

pp = pprint.PrettyPrinter(indent=4)

parser = ArgumentParser()
parser.add_argument('--log', help='Log all experiments with Sacred',
                    action='store_true')
args = parser.parse_args()

if not args.log:
    ex.observers.clear()
    print('Running without Sacred observers')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {'dataset_str': None,
          'encoder_str': None,
          'repr_str': None,
          'loss_str': 'square_exponential',
          'sampling_str': 'ranked',
          'dimensions': [256, 128],
          'edge_score': 'inner',
          'lr': 1e-3,
          'epochs': 200,
          'p_labeled': 0.1,
          'n_exper': 20,
          'device': device,
          'timestamp': str(int(datetime.now().timestamp()))}

# Values to be changed in experiments
param_grid = {'encoder_str': ['mlp', 'gcnmlp'],
              'repr_str': ['gaussian', 'euclidean_distance'],
              'dataset_str': ['cora', 'citeseer', 'pubmed', 'corafull',
                              'coauthorcs', 'coauthorphys',
                              'amazoncomp', 'amazonphoto']}

grid = ParameterGrid(param_grid)

for command in ['link_pred_experiments', 'node_class_experiments']:
    for i, hparams in enumerate(grid):
        print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
        config.update(hparams)

        pp.pprint(config)
        ex.run(command_name=command, config_updates=config)
