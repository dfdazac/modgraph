import torch
from train import ex
from sklearn.model_selection import ParameterGrid

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Configuration template
config = {'model_name': None,
          'device': device,
          'dataset': None,
          'hidden_dims': None,
          'lr': None,
          'epochs': 200,
          'random_splits': True,
          'rec_weight': 0}

# Values to be changed in experiments
param_grid = {'model_name': ['gae'],
              'dataset': ('cora', 'citeseer', 'pubmed', 'corafull',
                          'coauthorcs', 'coauthorphys', 'amazoncomp',
                          'amazonphoto'),
              'hidden_dims': ([256, 128],)}

grid = ParameterGrid(param_grid)

for i, hparams in enumerate(grid):
    print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
    config.update(hparams)
    ex.run(config_updates=config)
