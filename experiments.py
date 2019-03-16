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
          'encoder_str': 'sgc',
          'hidden_dims': [256, 128],
          'lr': 1e-4,
          'epochs': 200,
          'p_labeled': None,
          'n_exper': 20,
          'device': device,
          'timestamp': str(int(time.time())),
          'edge_score': 'inner'}

# Values to be changed in experiments
param_grid = {'dataset_str': ['cora', 'citeseer', 'pubmed'],
              'p_labeled': [0.02, 0.04, 0.06, 0.08, 0.1]}

grid = ParameterGrid(param_grid)

for task in ['link_pred_experiments', 'node_class_experiments']:
    for i, hparams in enumerate(grid):
        print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
        config.update(hparams)
        ex.run(command_name=task, config_updates=config)
