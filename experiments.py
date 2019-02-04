import torch
from train import ex
from sklearn.model_selection import ParameterGrid

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration template
config = {'model_name': None,
          'device': device,
          'dataset': None,
          'hidden_dims': [256, 128],
          'lr': 0.001,
          'epochs': 200,
          'random_splits': True,
          'rec_weight': 0,
          'encoder': 'gcn',
          'train_examples_per_class': 20,
          'val_examples_per_class': 30}

# Values to be changed in experiments
param_grid = {'model_name': ['graph2gauss'],
              'dataset': ('cora', 'citeseer', 'pubmed', 'corafull',
                          'coauthorcs', 'coauthorphys', 'amazoncomp',
                          'amazonphoto'),
              'lr': [0.005, 0.001, 0.0001]}

grid = ParameterGrid(param_grid)

for i, hparams in enumerate(grid):
    print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
    config.update(hparams)
    ex.run(config_updates=config)
