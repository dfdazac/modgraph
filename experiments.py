import torch
from train import ex
from sklearn.model_selection import ParameterGrid

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration template
config = {'model_name': None,
          'device': device,
          'dataset': None,
          'hidden_dims': None,
          'lr': None,
          'epochs': 2,
          'patience': 20,
          'random_splits': True}

# Values to be changed in experiments
param_grid = {'model_name': ('gae', 'dgi'),
              'dataset': ('cora', 'citeseer', 'pubmed'),
              'hidden_dims': ([32, 16], [128, 64], [512, 128]),
              'lr': (0.01, 0.005, 0.001)}

grid = ParameterGrid(param_grid)

for i, hparams in enumerate(grid):
    config.update(hparams)
    if config['dataset'] == 'pubmed':
        config['device'] = 'cpu'
    else:
        config['device'] = device
    
    ex.run(config_updates=config)
