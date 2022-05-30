import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

def get_loss(loss_type, **kwargs):
    if loss_type == 'CE':
        criterion = CrossEntropy(**kwargs)
    elif loss_type == 'FCE':
        criterion = FocalCrossEntropy(**kwargs)
    elif loss_type == 'ICE':
        criterion = InverseFrequencyCrossEntropy(**kwargs)
    else:
        raise
    return criterion

def label2target(pred: torch.Tensor, label: List):
    target = torch.zeros_like(pred)
    for i, lb in enumerate(label):
        target[i, lb] = 1
    return target

class _Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction

class _WeightedLoss(_Loss):
    """
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
    """
    def __init__(self, weight = None, ignore_index=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(reduction)
        assert reduction in ['mean', 'sum', 'none']
        self.ignore_index = ignore_index
        self.weight = weight
        if self.weight is not None and self.ignore_index is not None:
            assert isinstance(ignore_index, int)
            self.weight[ignore_index] = 0

# Cross Entropy
class CrossEntropy(_WeightedLoss):
    """
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
    Return:
        a float value when reduction is 'mean' or 'sum'.
        a list of float values when reduction is 'none'.
    """
    def __init__(self, weight=None, ignore_index=None, reduction='mean'):
        super(CrossEntropy, self).__init__(weight, ignore_index, reduction)
        
    def forward(self, predict, target, argmax=False, softmax=True):
        assert predict.shape == target.shape, f'predict & target shape do not match. {predict.shape} != {target.shape}'
        if softmax:
            predict = F.softmax(predict, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        predict = predict.contiguous()
        target = target.contiguous()
            
        loss = self.get_loss(predict, target) # N, C
        if self.weight is not None:
            loss = torch.mul(self.weight, loss)
        loss = loss.sum(dim=1) # N
        loss *= -1
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
    
    def get_loss(self, predict, target):
        loss = torch.mul(target, torch.log(predict + 1e-9))
        return loss
    
class FocalCrossEntropy(CrossEntropy):
    def __init__(self, weight=None, ignore_index=None, alpha=1, gamma=2, reduction='mean'):
        super(FocalCrossEntropy, self).__init__(weight, ignore_index, reduction)
        self.alpha = alpha
        self.gamma = gamma
    
    def get_loss(self, predict, target):
        loss = (1-predict).pow(self.gamma).mul(self.alpha).mul(target).mul(torch.log(predict + 1e-9))
        return loss
    
class InverseFrequencyCrossEntropy(_WeightedLoss):
    """
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
    Return:
        a float value when reduction is 'mean' or 'sum'.
        a list of float values when reduction is 'none'.
    """
    def __init__(self, weight=None, ignore_index=None, reduction='mean'):
        super(InverseFrequencyCrossEntropy, self).__init__(weight, ignore_index, reduction)
        
    def forward(self, predict, target, argmax=False, softmax=True):
        assert predict.shape == target.shape, f'predict & target shape do not match. {predict.shape} != {target.shape}'
        if softmax:
            predict = F.softmax(predict, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        predict = predict.contiguous()
        target = target.contiguous()
            
        loss = self.get_loss(predict, target) # N, C
        if self.weight is not None:
            loss = torch.mul(self.weight, loss)
        loss = loss.sum(dim=1) # N
        loss *= -1
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
    
    def get_loss(self, predict, target, power=1, eps=1e-9):
        _weights = 1.0/(target.sum(0).pow(power)+eps) # C
        weights = _weights/_weights.sum()*225 # C
        loss = torch.mul(target, weights).mul(torch.log(predict+eps))
        return loss