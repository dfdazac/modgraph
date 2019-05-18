## A Modular Framework for Unsupervised Graph Representation Learning

Methods for unsupervised representation learning on graphs can be described in terms of modules:

- Graph encoders
- Representations
- Scoring functions
- Loss functions
- Sampling strategies

By identifying this we can reproduce existing methods:

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


### Instructions

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

To train GAE and evaluate it in the link prediction task using the Cora dataset, run

```sh
python train.py with dataset_str='cora' \
encoder_str='gcn' \
repr_str='euclidean_inner' \
loss_str='bce_loss' \
sampling_str='first_neighbors' \
n_exper=1
```

