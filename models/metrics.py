import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, f1_score, accuracy_score
import numpy as np

class AverageMeter(object):
    """Compute metrics and update value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def logloss(target, pred):
    pred = torch.sigmoid(pred)
    pred = pred.detach().cpu().numpy()
    # print("pred shapeeeeeeeeee:", pred.shape)
    target = target.detach().cpu().numpy()
    # print("target shapeeeeeee: ", target.shape)
    return log_loss(target, pred, normalize=False)

def f1score(target, pred):
    
    pred = pred.detach().cpu().clone().numpy()
    target = target.detach().cpu().clone().numpy()
    return f1_score(target, pred)

def acc(target, pred, thresh=0.5):
    # pred = (torch.sigmoid(pred)>thresh).type(target.type())
    pred = torch.argmax(pred, )
    
    pred = pred.detach().cpu().clone().numpy()
    
    target = target.detach().cpu().clone().numpy()
    return accuracy_score(target, pred)


if __name__=="__main__":
    pass