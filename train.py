from datetime import datetime
import os.path as osp
from argparse import Namespace

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorboardX import SummaryWriter
from sacred import Experiment
from sacred.observers import MongoObserver

from utils import split_edges, add_reverse_edges
from models import GAE, DGI, NodeClassifier


def eval_link_prediction(emb, edges_pos, edges_neg):
    """Evaluate the AUC and AP scores when using the provided embeddings to
    predict links between nodes.

        - emb: tensor of shape (N, d) where N is the number of nodes and d
            the dimension of the embeddings.
        - edges_pos, edges_neg: tensors of shape (2, p) containing positive
        and negative edges, respectively, in their columns.
    Returns:
        - auc_score, float
        - ap_score, float
    """
    # Get scores for edges using inner product
    pos_score = (emb[edges_pos[0]] * emb[edges_pos[1]]).sum(dim=1)
    neg_score = (emb[edges_neg[0]] * emb[edges_neg[1]]).sum(dim=1)
    preds = torch.cat((pos_score, neg_score)).cpu().numpy()

    targets = torch.cat((torch.ones_like(pos_score),
                         torch.zeros_like(neg_score))).cpu().numpy()

    auc_score = roc_auc_score(targets, preds)
    ap_score = average_precision_score(targets, preds)

    return auc_score, ap_score


def train_encoder(model_name, device, dataset, hidden_dims, lr, epochs,
                  patience, random_splits):
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    if not torch.cuda.is_available() and device.startswith('cuda'):
        raise ValueError(f'Device {device} specified '
                         'but CUDA is not available')

    device = torch.device(device)

    # Load data
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    dataset = Planetoid(path, dataset)
    data = dataset[0]
    # Obtain edges for the link prediction task
    positive_splits, negative_splits = split_edges(data.edge_index)
    train_pos, val_pos, test_pos = positive_splits
    train_neg, val_neg, test_neg = negative_splits
    # Add edges in reverse direction for encoding
    train_pos = add_reverse_edges(train_pos).to(device)
    train_neg = add_reverse_edges(train_neg).to(device)

    # Create model
    if model_name == 'dgi':
        model_class = DGI
    elif model_name == 'gae':
        model_class = GAE
    else:
        raise ValueError(f'Unknown model {model_name}')

    model = model_class(dataset.num_features, hidden_dims).to(device)

    # Train model
    logdir = osp.join('runs', f'{now}')
    ckpt_path = osp.join(logdir, 'checkpoint.p')
    writer = SummaryWriter(logdir)
    print(f'Training {model_name}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc = 0
    patience_count = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = model(data, train_pos, train_neg)
        loss.backward()
        optimizer.step()

        # Evaluate on val edges
        embeddings = model.encoder(data, train_pos).cpu().detach()
        auc, ap = eval_link_prediction(embeddings, val_pos, val_neg)
        print('\r[{:03d}/{:03d}] train loss: {:.6f}, '
              'val_auc: {:6f}, val_ap: {:6f}'.format(epoch,
                                                      epochs,
                                                      loss.item(),
                                                      auc, ap),
              end='', flush=True)

        if auc > best_auc:
            # Keep best model on val set
            best_auc = auc
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            # Terminate early based on patience
            patience_count += 1
            if patience_count == patience:
                break

    # Evaluate on test edges
    model.load_state_dict(torch.load(osp.join(ckpt_path)))
    model.eval()
    embeddings = model.encoder(data, train_pos).cpu().detach()
    auc, ap = eval_link_prediction(embeddings, test_pos, test_neg)
    print('\ntest_auc: {:6f}, test_ap: {:6f}'.format(auc, ap))

    # Evaluate embeddings in node classification
    classifier = NodeClassifier(model.encoder,
                                hidden_dims[-1],
                                dataset.num_classes).to(device)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    if random_splits:
        # Generate new random masks
        num_labels_all = sum(map(lambda x: x[1].sum().item(),
                             data('train_mask', 'val_mask', 'test_mask')))
        mask_idx = np.random.choice(range(data.num_nodes), num_labels_all,
                                    replace=False)
        masks = []
        start = 0
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            new_mask = torch.zeros_like(mask)
            num_labels = torch.sum(mask).item()
            new_mask[mask_idx[start:start + num_labels]] = 1
            masks.append(new_mask)
            start += num_labels

        # Reassign label masks
        for i, (name, _) in enumerate(data('train_mask', 'val_mask', 'test_mask')):
            setattr(data, name, masks[i])

    def train_classifier():
        classifier.train()
        classifier_optimizer.zero_grad()
        output = classifier(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        classifier_optimizer.step()
        return loss.item()

    def test_classifier():
        classifier.eval()
        logits = classifier(data)
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    print('Training node classifier')
    best_accs = []
    best_val_acc = 0
    patience_count = 0
    epochs = 100
    for epoch in range(1, epochs + 1):
        train_classifier()
        accs = test_classifier()
        log = '\r[{:03d}/{:03d}] train: {:.4f}, val: {:.4f}, test: {:.4f}'
        print(log.format(epoch, epochs, *accs), end='', flush=True)

        val_acc = accs[1]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_accs = accs
            patience_count = 0
        else:
            # Terminate early based on patience
            patience_count += 1
            if patience_count == patience:
                break

    log = '\nBest validation results\nTrain: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(*best_accs))
    test_acc = best_accs[2]

    return auc, ap, test_acc


ex = Experiment()
ex.observers.append(MongoObserver.create(url='mongodb://daniel:daniel1@ds151814.mlab.com:51814/experiments',
                                         db_name='experiments'))


@ex.config
def config():
    model_name = 'gae'
    device = 'cpu'
    dataset = 'cora'
    hidden_dims = [32, 16]
    lr = 0.001
    epochs = 200
    patience = 20
    random_splits = True


@ex.automain
def run_experiments(model_name, device, dataset, hidden_dims, lr, epochs,
                    patience, random_splits, _run):
    torch.random.manual_seed(42)
    np.random.seed(42)
    n_exper = 10
    results = np.empty([n_exper, 3])

    for i in range(10):
        print('\nExperiment {:d}/{:d}'.format(i + 1, n_exper))
        results[i] = train_encoder(model_name, device, dataset, hidden_dims,
                                   lr, epochs, patience, random_splits)

    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    print('-'*50)
    print('Final results')
    metrics = ['AUC', 'AP', 'ACC']
    for i in range(3):
        print('{}: {:.1f} ± {:.2f}'.format(metrics[i],
                                           mean[i] * 100,
                                           std[i] * 100))
        _run.log_scalar(f'{metrics[i]} mean', mean[i])
        _run.log_scalar(f'{metrics[i]} std', std[i])