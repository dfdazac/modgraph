from argparse import ArgumentParser
import time

import torch
from train import ex
from sklearn.model_selection import ParameterGrid

parser = ArgumentParser()
parser.add_argument('--log', help='Log all experiments with Sacred',
                    action='store_true')
args = parser.parse_args()

if not args.log:
    ex.observers.clear()
    print('Running without Sacred observers')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration template
config = {'dataset_str': None,
          'method': 'gae',
          'encoder_str': None,
          'hidden_dims': [256, 128],
          'rec_weight': 0,
          'lr': None,
          'epochs': 1,
          'p_labeled': 0.1,
          'n_exper': 20,
          'device': device,
          'timestamp': str(int(time.time()))}

# Values to be changed in experiments
param_grid = {'encoder_str': ['gcn', 'mlp'],
              'dataset_str': ['cora', 'citeseer', 'pubmed', 'corafull',
                              'coauthorcs', 'coauthorphys', 'amazoncomp',
                              'amazonphoto'],
              'lr': [1e-3, 1e-4, 5e-5]}

grid = ParameterGrid(param_grid)

for task in ['node_class_experiments']:
    for i, hparams in enumerate(grid):
        print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
        config.update(hparams)
        ex.run(command_name=task, config_updates=config)
