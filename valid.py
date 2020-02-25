import apex
from apex import amp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from pytorch_toolbelt.inference import tta as pytta
import time

from models.meta_modell import ResNet, EfficientNet, DenseNet
from models.metrics import logloss,f1score, AverageMeter
from models.utils import save_checkpoint, mixup_data, cutmix_data, accuracy
from sklearn.metrics import log_loss, f1_score, accuracy_score
from config import get_cfg_defaults
import warnings
warnings.filterwarnings("ignore")


def valid_model(cfg, model,valid_loader, weight, tta=False):


    model.eval()
    tbar = tqdm(valid_loader)
    preds = []
    targets = []
    top1 = AverageMeter()
    with torch.no_grad():
        for i, (image, target, onehot) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            onehot = onehot.cuda()

            output = model(image)
            
            
            # pred = torch.sigmoid(output)
            pred = torch.softmax(output, dim=1)
            preds.append(pred.cpu())
            targets.append(target.cpu())
            
            #Accuracy
            acc = accuracy(onehot, pred)
            
            top1.update(acc, image.size(0))
            
    #convert this shit to numpy
    preds, targets = torch.cat(preds, 0), torch.cat(targets, 0)
    preds = preds.numpy()
    targets = targets.numpy()

    #compute fucking metrics
    lloss = log_loss(targets, preds, eps=1e-7, labels=np.array([0.,1.,2.]))
    # lloss = np.mean(lloss)


    print(f"Valid logloss: {lloss}, acc: {top1.avg}")

if __name__ == "__main__":
    pass