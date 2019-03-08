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
          'method': None,
          'encoder_str': 'sgc',
          'hidden_dims': [256, 128],
          'lr': None,
          'epochs': 200,
          'p_labeled': 0.1,
          'n_exper': 20,
          'device': device,
          'timestamp': str(int(time.time())),
          'edge_score': 'inner'}

# Values to be changed in experiments
param_grid = {'method': ['gae', 'dgi'],
              'dataset_str': ['cora', 'citeseer', 'pubmed', 'corafull',
                              'coauthorcs', 'coauthorphys', 'amazoncomp',
                              'amazonphoto'],
              'lr': [1e-3, 1e-4, 1e-5]}

grid = ParameterGrid(param_grid)

for task in ['link_pred_experiments', 'node_class_experiments']:
    for i, hparams in enumerate(grid):
        print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
        config.update(hparams)
        ex.run(command_name=task, config_updates=config)
