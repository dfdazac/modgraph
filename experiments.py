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
          'epochs': 200,
          'random_splits': True}

# Values to be changed in experiments
param_grid = {'model_name': ('gae', 'dgi'),
              'dataset': ('corafull'),
              'hidden_dims': ([256, 128],),
              'lr': (0.01, 0.005, 0.001)}

grid = ParameterGrid(param_grid)

for i, hparams in enumerate(grid):
    print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
    config.update(hparams)
    ex.run(config_updates=config)
