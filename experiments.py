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
          'rec_weight': None,
          'encoder': 'gcn',
          'train_examples_per_class': 20,
          'val_examples_per_class': 30,
          'n_exper': 20}

# Values to be changed in experiments
param_grid = {'model_name': ['gae'],
              'dataset': ('cora', 'citeseer', 'pubmed', 'corafull',
                          'coauthorcs', 'coauthorphys', 'amazoncomp',
                          'amazonphoto'),
              'rec_weight': [0.1, 0.5, 0.9]}

grid = ParameterGrid(param_grid)

for i, hparams in enumerate(grid):
    print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
    config.update(hparams)
    ex.run(config_updates=config)
