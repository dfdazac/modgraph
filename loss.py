import torch
import torch.nn.functional as F


def bce_loss(pos_score, neg_score):
    preds = torch.cat((pos_score, neg_score))
    targets = torch.cat((torch.ones_like(pos_score),
                         torch.zeros_like(neg_score)))
    return F.binary_cross_entropy(preds, targets)


def square_exponential(pos_score, neg_score):
    return (torch.log(pos_score) ** 2 + torch.exp(neg_score)).mean()
