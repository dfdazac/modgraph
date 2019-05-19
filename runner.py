import os.path as osp
import os
from datetime import datetime
import sherpa

parameters = [sherpa.Choice('encoder_str',
                            range=['mlp', 'gcn', 'sgc', 'gcnmlp']),
              sherpa.Choice('repr_str',
                            range=['euclidean_inner',
                                   'euclidean_bilinear',
                                   'euclidean_distance',
                                   'gaussian']),
              sherpa.Choice('loss_str',
                            range=['bce_loss',
                                   'square_exponential_loss',
                                   'square_square_loss',
                                   'hinge_loss']),
              sherpa.Choice('sampling_str',
                            range=['first_neighbors',
                                   'graph_corruption',
                                   'ranked'])]

timestamp = str(int(datetime.now().timestamp()))
output_dir = osp.join('./logs', timestamp)
if not osp.isdir(output_dir):
    os.mkdir(output_dir)

command = f'train.py -u parallel_trial with n_exper=1'

algorithm = sherpa.algorithms.GridSearch()
scheduler = sherpa.schedulers.LocalScheduler()
results = sherpa.optimize(parameters=parameters,
                          algorithm=algorithm,
                          lower_is_better=False,
                          filename=command,
                          output_dir=output_dir,
                          scheduler=scheduler,
                          max_concurrent=4,
                          verbose=1)
