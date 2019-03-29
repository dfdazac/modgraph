

**Instructions**

Create a conda environment with all the requirements (edit `environment.yml` if you want to change the name of the environment):

```sh
conda env create -f environment.yml
```

Activate the environment

```sh
source activate graphlearn
```

We use [Sacred](https://github.com/IDSIA/sacred) to run and log all the experiments. To list the configuration variables and their default values, run

```sh
python train.py print_config
```

Two commands are available: `link_pred_experiments` and `node_class_experiments`.

**Example**

To train a model with a GCN encoder, the GAE objective, and evaluate it in the node classification task using the Cora dataset, run

```sh
python train.py node_class_experiments with dataset_str='cora' encoder_str='gcn' method='gae' n_exper=1
```

