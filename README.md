## A Modular Framework for Unsupervised Graph Representation Learning

Methods for unsupervised representation learning on graphs can be described in terms of modules:

- Graph encoders
- Representations
- Scoring functions
- Loss functions
- Sampling strategies

By identifying this we can reproduce existing methods:

**Variational Graph Autoencoders** [(Kipf and Welling, 2016)](https://arxiv.org/abs/1611.07308):

```python
encoder = GCNEncoder(dataset.num_features, hidden_dims=[256, 128])
representation = GaussianVariational()
loss = bceloss
sampling = FirstNeighborSampling
```

**Graph Autoencoders** [(Kipf and Welling, 2016)](https://arxiv.org/abs/1611.07308):

```python
encoder = GCNEncoder(dataset.num_features, hidden_dims=[256, 128])
representation = EuclideanInnerProduct()
loss = bceloss
sampling = FirstNeighborSampling
```

**Deep Graph Infomax** [(Veličković et al., 2018)](https://arxiv.org/abs/1809.10341):

```python
encoder = GCNEncoder(dataset.num_features, hidden_dims=[256, 128])
representation = EuclideanBilinear()
loss = bceloss
sampling = GraphCorruptionSampling
```

**Graph2Gauss** [(Bojchevski and Günnemann, 2017)](https://arxiv.org/abs/1707.03815):

```python
encoder = MLPEncoder(dataset.num_features, hidden_dims=[256, 128])
representation = Gaussian()
loss = square_exponential
sampling = RankedSampling
```

We can also use this framework to create new methods. For example, we can simplify Graph2Gauss with an Euclidean distance:

```python
encoder = MLPEncoder(dataset.num_features, hidden_dims=[256, 128])
representation = EuclideanDistance()
loss = square_exponential
sampling = RankedSampling
```

Under this framework, all these methods can be trained and evaluated with the same procedure:

```python
method = EmbeddingMethod(encoder, representation, loss, sampling)
embeddings, results = train(dataset, method)
```


### Installation

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

## Running the experiments

The default settings train our best method (EB-GAE) on the link prediction task with the Cora dataset:

```sh
python train.py link_pred_experiments
```

Other methods can be evaluated as well:

GAE

```sh
python train.py link_pred_experiments \
    with dataset_str='cora' \
    encoder_str='gcn' \
    repr_str='euclidean_inner' \
    loss_str='bce_loss' \
    sampling_str='first_neighbors'
```

DGI

```sh
python train.py link_pred_experiments \
    with dataset_str='cora' \
    encoder_str='gcn' \
    repr_str='euclidean_infomax' \
    loss_str='bce_loss' \
    sampling_str='graph_corruption'
```

Graph2Gauss

```sh
python train.py link_pred_experiments \
    with dataset_str='cora' \
    encoder_str='mlp' \
    repr_str='gaussian' \
    loss_str='square_exponential_loss' \
    sampling_str='ranked'
```