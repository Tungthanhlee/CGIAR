
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
from .resnet import ResNet
from .meta_modell import ResNet, EfficientNet, DenseNet
from .losses import RandomLabelSmoothingLoss
from .metrics import logloss,f1score, AverageMeter
from .utils import save_checkpoint, mixup_data, cutmix_data, accuracy
from sklearn.metrics import log_loss, f1_score, accuracy_score
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from torchcontrib.optim import SWA

import warnings
warnings.filterwarnings("ignore")

def get_model(cfg):
    if 'efficientnet' in cfg.TRAIN.MODEL:
        model = EfficientNet(cfg)
    elif 'res' in cfg.TRAIN.MODEL:
        model = ResNet(cfg)
    elif "dense" in cfg.MODEL.NAME:
        model = DenseNet(cfg)
    return model



def test_model(_print, cfg, model, test_loader, weight, tta=False):
    
    model.load_state_dict(torch.load(weight)["state_dict"])
    if tta:
        print("#############TTA###############")
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)

    
    model.eval()
    tbar = tqdm(test_loader)
    preds = []
    ids = []
    with torch.no_grad():
        for i, (image, _id) in enumerate(tbar):
            image = image.cuda()      
            output = model(image)
            
            pred = torch.softmax(output, dim=1)
            preds.append(pred.cpu())
            for j in _id:
                ids.append(j)
    preds = torch.cat(preds, 0).numpy()
    ids = np.asarray(ids)
    # print(preds.shape)
    
   
    np.save(os.path.join(cfg.DIRS.SUB, cfg.EXP + f"_fold{cfg.DATA.FOLD}.npy"), preds)
    np.save(os.path.join(cfg.DIRS.SUB, "names.npy"), ids)
    # print(ids.shape)

    


def valid_model(_print, cfg, model, valid_criterion,valid_loader, tta=False):

    if tta:
        print("#############TTA###############")
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)

    model.eval()
    tbar = tqdm(valid_loader)
    preds = []
    targets = []
    
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.no_grad():
        for i, (image, target, onehot) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            onehot = onehot.cuda()

            output = model(image)
            loss = valid_criterion(output, target)
            losses.update(loss.item(), image.size(0))
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


    _print("Valid logloss: %.3f, acc: %.3f, CEloss: %.3f" % (lloss, top1.avg, losses.avg))

    return lloss



def train_loop(_print, cfg, model, train_loader,valid_loader, criterion, valid_criterion, optimizer, scheduler, start_epoch, best_metric):
    time_all = time.time()
    if cfg.DEBUG == False:
        #{cfg.EXP}_{cfg.TRAIN.MODEL}_fold{cfg.DATA.FOLD}
        tb = SummaryWriter(f"runs/{cfg.EXP}_{cfg.TRAIN.MODEL}_fold{cfg.DATA.FOLD}") #for visualization
    """
    TRAIN
    """
    
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        
        time_ep = time.time()
        
        if cfg.MODEL.RANDOM_LS:
            train_criterion = RandomLabelSmoothingLoss(smoothing=np.random.uniform(low=0.01, high=0.3))
            train_criterion = train_criterion.cuda()
        _print(f"Epoch {epoch + 1}")

        losses = AverageMeter()
        model.train()
        tbar = tqdm(train_loader)

        for i, (image, target, _) in enumerate(tbar):
            # print(target)          
            image = image.cuda()
            target = target.cuda()
            # mixup/ cutmix
            if cfg.DATA.MIXUP:
                image = mixup_data(image, alpha=cfg.DATA.CM_ALPHA)
            elif cfg.DATA.CUTMIX:
                image = cutmix_data(image, alpha=cfg.DATA.CM_ALPHA)
            output = model(image)
            
            if cfg.MODEL.RANDOM_LS:
                loss = train_criterion(output, target)
            else:
                loss = criterion(output, target)

            # gradient accumulation
            loss = loss / cfg.OPT.GD_STEPS

            if cfg.SYSTEM.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                if cfg.OPT.CLR:
                    scheduler(optimizer, i, epoch)
                else:
                    scheduler(optimizer, i, epoch, None) # Cosine LR Scheduler
                optimizer.step()
                optimizer.zero_grad()

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            tbar.set_description("Train loss: %.3f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))

            if cfg.DEBUG == False:
                #tensorboard
                tb.add_scalars('Loss', {'loss':losses.avg}, epoch)
                # tb.add_scalars('Train',
                #             {'top_lloss':top_lloss.avg}, epoch)
                tb.add_scalars('Lr', {'Lr':optimizer.param_groups[-1]['lr']}, epoch)


        _print("Train loss: %.3f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))

        """
        VALID
        """
        top_lloss = valid_model(_print, cfg, model, valid_criterion, valid_loader, tta = cfg.INFER.TTA)
        # is_best = top_lloss > best_metric
        # best_metric = max(top_lloss, best_metric)
        is_best = top_lloss < best_metric
        best_metric = min(top_lloss, best_metric)
        _print("Current best: %.3f" % best_metric)
        #time
        time_ep = time.time() - time_ep
        _print(f"Time per epoch: {round(time_ep,4)} seconds")

        #tensorboard
        if cfg.DEBUG == False:
            tb.add_scalars('Valid',
                            {'top_lloss':top_lloss}, epoch)
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": cfg.EXP,
                "state_dict": model.state_dict(),
                "best_metric": best_metric,
                "optimizer": optimizer.state_dict(),
            }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}_{cfg.TRAIN.MODEL}_fold{cfg.DATA.FOLD}.pth")

    #time
    time_all = (time.time() - time_all)/60.
    _print(f"Time all: {round(time_all,4)} mins")

    if cfg.DEBUG == False:
        # tb.export_scalars_to_json(os.path.join(cfg.DIRS.OUTPUTS, f"{cfg.EXP}_{cfg.TRAIN.MODEL}_fold{cfg.DATA.FOLD}_{round(best_metric,4)}.json"))
        tb.close()


def swa_train_loop(_print, cfg, model, swa_model, train_loader,valid_loader, criterion, valid_criterion, optimizer, scheduler, epoch, best_metric):
    
    swa_n = 0
    
    
    train_loop(_print, cfg, model, \
            train_loader, criterion,\
            optimizer, scheduler, epoch)
    # current_metric = valid_model(_print, cfg, model, valid_criterion, valid_loader, best_metric, optimizer, epoch, tta=cfg.INFER.TTA)
    
    swa_start = int(cfg.TRAIN.EPOCHS*0.75)
    valid_model(_print, cfg, model, valid_criterion, valid_loader, best_metric, optimizer, epoch, tta=cfg.INFER.TTA)
    
    if (epoch+1) > swa_start and (epoch+1 - swa_start) % cfg.MODEL.SWA_CIRCLES == 0:
        moving_average(swa_model, model, 1.0/(swa_n+1))
        swa_n +=1
        if (epoch == 0) or (epoch % cfg.MODEL.SWA_EVAL_FREQ == cfg.MODEL.SWA_EVAL_FREQ - 1) or (epoch == cfg.TRAIN.EPOCHS-1):
            bn_update(train_loader, swa_model)
            valid_model(_print, cfg, swa_model, valid_criterion, valid_loader, best_metric, optimizer, epoch, tta=cfg.INFER.TTA)
    # _print("Current best metrics: %.3f" % (round(best, 4)))
    # if (epoch+1) == cfg.MODEL.SWA_START:
    #     if current_metric < cfg.MODEL.SWA_THRESHOLD:
    #         # print("@@@@@@@@@@@", current_metric)
    #         copy_model(swa_model, model)
    #         swa_n = 0
    #     else:
    #         delay_swa = 1
    # if ((epoch+1) >= cfg.MODEL.SWA_START) and ((epoch+1) % cfg.MODEL.SWA_FREQ == 0) and not delay_swa:
    #     if current_metric < cfg.MODEL.SWA_THRESHOLD:
    #         moving_average(swa_model, model, 1.0 / (swa_n + 1))
    #         swa_n += 1
    #         bn_update(train_loader, swa_model)
    #         current_metric = valid_model(_print, cfg, model, valid_criterion, valid_loader, best_metric, optimizer, epoch, tta=cfg.INFER.TTA)
    #     else:
    #         delay_swa = 1
    # if ((epoch+1) >= cfg.MODEL.SWA_START) and delay_swa:
    #     if current_metric < cfg.MODEL.SWA_THRESHOLD:
    #         # delay_swa = 0
    #         if swa_n == -1:
    #             copy_model(swa_model, model)
    #             swa_n = 0
    #         else:
    #             moving_average(swa_model, model, 1.0 / (swa_n + 1))
    #             swa_n += 1
    #         bn_update(train_loader, swa_model)
    #         current_metric = valid_model(_print, cfg, model, valid_criterion, valid_loader, best_metric, optimizer, epoch, tta=cfg.INFER.TTA)
    

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def copy_model(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 0
        param1.data += param2.data

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    tbar = tqdm(loader)
    for i, (input, _) in enumerate(tbar):
        input = input.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))