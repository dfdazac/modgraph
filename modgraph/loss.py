import torch
import torch.nn.functional as F


def bce_loss(pos_score, neg_score):
    preds = torch.cat((pos_score, neg_score))
    targets = torch.cat((torch.ones_like(pos_score),
                         torch.zeros_like(neg_score)))
    return F.binary_cross_entropy_with_logits(preds, targets)


def square_exponential_loss(pos_score, neg_score):
    return (pos_score ** 2 + torch.exp(neg_score)).mean()


def square_square_loss(pos_score, neg_score):
    margin_diff = 1.0 + neg_score
    margin_diff[margin_diff < 0] = 0.0
    loss = torch.mean(pos_score ** 2) + torch.mean(margin_diff ** 2)
    return loss


def hinge_loss(pos_score, neg_score):
    margin = 1.0 - pos_score + neg_score
    margin[margin < 0] = 0
    loss = margin.mean()
    return loss
