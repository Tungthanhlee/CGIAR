import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from config import get_cfg_defaults
import datasets

cfg = get_cfg_defaults()
dl_train = datasets.get_dataset(cfg, mode = "train")
dl_val = datasets.get_dataset(cfg, mode = "valid")

class ResNet(nn.Module):

    def __init__(self, model_name, num_classes):
        super(ResNet, self).__init__()

        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained="imagenet")
        in_features = model.last_linear.in_features
        del model.last_linear
        feature_map = list(model.children())
        self.backbone = nn.Sequential(*list(feature_map))

        self.fc = nn.Linear(in_features, num_classes)
    
    def features(self, x):
        return self.backbone(x)
    
    def logits(self, x):
        return self.fc(x)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.logits(x)
        return x

images, labels=next(iter(dl_train))


model = ResNet(model_name='resnet50', num_classes=3)
in_ = torch.rand((1, 3,244,244))
out = model(images[0:1])
print(out.shape)