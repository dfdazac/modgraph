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

config = {'dataset_str': 'cora',
          'method': 'sge',
          'encoder_str': None,
          'hidden_dims': [256, 128],
          'n_points': None,
          'lr': None,
          'epochs': 200,
          'p_labeled': 0.1,
          'n_exper': 1,
          'device': device,
          'timestamp': str(int(datetime.now().timestamp())),
          'edge_score': 'inner'}

# Values to be changed in experiments
param_grid = {'encoder_str': ['mlp', 'gcn', 'sgc'],
              'n_points': [1, 4, 8, 16, 32, 64],
              'lr': [1e-2, 1e-3, 1e-4]}

grid = ParameterGrid(param_grid)

for i, hparams in enumerate(grid):
    print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
    config.update(hparams)

    pp.pprint(config)
    ex.run(command_name='link_pred_experiments', config_updates=config)
