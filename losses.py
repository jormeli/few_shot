import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

@gin.configurable(blacklist=['y_pred', 'y_true'])
def triplet_loss(y_pred, y_true, margin=0.5):
    s = y_pred.size()
    x = y_pred.reshape(-1, 3, s[-1])
    anchor = x[:, 0]
    positive = x[:, 1]
    negative = x[:, 2]
    d1 = F.mse_loss(anchor, positive, reduction='none').sum(1)
    d2 = F.mse_loss(anchor, negative, reduction='none').sum(1)

    zeros = torch.zeros(d1.size(0))
    zeros = zeros.to(torch.device(d1.device))
    loss = torch.max(d1 - d2 + margin, zeros).mean()

    return loss

@gin.configurable(blacklist=['y_pred', 'y_true'])
def dummy_loss(y_pred, y_true):
    return torch.zeros(1)
