import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
cv2.setNumThreads(0)
# cv2.oc1.setUseOpenCL(False)
import torch.nn.functional as F
# from albumentations import *
# from albumentations.pytorch import ToTensor
from config import get_cfg_defaults
from torch.autograd import Variable
# from torchvision import transforms
from PIL import Image
import torchvision
from torchvision.transforms import transforms
from augment import RandAugment
# from RandAugment import RandAugment
import warnings
warnings.filterwarnings("ignore")


class CGIAR(Dataset):

    def __init__(self, cfg, csv, mode):
        super(CGIAR, self).__init__()
        
        self.cfg = cfg
        self.mode = mode
        self.df = pd.read_csv(csv)  
        
        # self.data_root = os.path.join(cfg.DIRS.DATA, "train") 
        # if self.mode == 'test':
        #     self.data_root = os.path.join(cfg.DIRS.DATA, "test") 
        
        self.data_root = os.path.join(cfg.DIRS.DATA, "bunch") 
        self.size = (cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
        self.resize_crop = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(self.size,
                                                            scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)])
        if self.mode == "train":
            self.transform = RandAugment(n=cfg.TRAIN.RANDAUG_N, m=cfg.TRAIN.RANDAUG_M)
            
    def __len__(self):
        return len(self.df)

    def _load_img(self, img_path):
        """
        Input: Take image path
        Output: Return image as an array
        """
        
        try:
            image = Image.open(img_path)
        except:
            try:
                image = Image.open(img_path+'.jpg')
            except:
                try:
                    image = Image.open(img_path+'.JPG')
                except:
                    image = Image.open(img_path+'.jfif')
        # image = Image.open(img_path)
        image = image.convert('RGB')   
        return image

    def __getitem__(self, idx):
        
        info = self.df.loc[idx]
        img_path = os.path.join(self.data_root, info["id"])
        image = self._load_img(img_path) #load img
        
        if self.mode == "train" and self.cfg.TRAIN.AUG == True:
            image = self.transform(image)

        # image = self.resize_crop(image) #resize image
        image = image.resize(self.size)
        
        #convert from PIL image to np array
        image = torch.from_numpy(np.asarray(image)).float() 
        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2,0,1)
        image = image.div_(255.)
                

        if self.mode == "test":
            _id = info["id"]
            return image, _id
        else: 
            #get class
            class_ = info["label"]
            class_ = np.asarray(class_)
            label = torch.from_numpy(class_).type(image.type())
            label = torch.tensor(label, dtype=torch.long)
            return image, label

def get_dataset(cfg, mode):
    
    if mode == 'train':
        if cfg.DATA.PSEUDO == True:
            csv = os.path.join(cfg.DATA.CSV,f"pseudo.csv")
        else:
            csv = os.path.join(cfg.DATA.CSV,f"train_fold{cfg.DATA.FOLD}.csv")
        dts = CGIAR(cfg, csv, mode)
        batch_size = cfg.TRAIN.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=True, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    elif mode == 'valid' or mode == 'test':
        if mode == 'test':
            csv = os.path.join(cfg.DATA.CSV,"test.csv")
        else:
            csv = os.path.join(cfg.DATA.CSV,f"valid_fold{cfg.DATA.FOLD}.csv")
        dts = CGIAR(cfg,csv, mode)
        batch_size = cfg.VAL.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader

def get_debug_dataset(cfg, mode):
    # cfg = get_cfg_defaults()
    if mode == 'train':
        csv = os.path.join(cfg.DATA.CSV,f"train_fold{cfg.DATA.FOLD}.csv")
        dts = CGIAR(cfg, csv, mode)
        dts = Subset(dts, np.random.choice(np.arange(len(dts)), 5))
        # dts = Subset(dts)
        batch_size = cfg.TRAIN.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=True, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    elif mode == 'valid':
        csv = os.path.join(cfg.DATA.CSV,f"valid_fold{cfg.DATA.FOLD}.csv")
        dts = CGIAR(cfg,csv, mode)
        dts = Subset(dts, np.random.choice(np.arange(len(dts)), 2))
        batch_size = cfg.VAL.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader



if __name__ == "__main__":
    cfg = get_cfg_defaults()
    # csv = os.path.join(cfg.DATA.CSV,f"train_fold{cfg.DATA.FOLD}.csv")
    # dts = CGIAR(cfg,csv, mode="train")
    # print("len: ",dts.__len__())
    # img, label = dts.__getitem__(500)
    # print(img.shape, label)
    # dl = get_dataset(cfg, mode = "train")
    # print(dl)
    img = torch.rand((3,224,224))
    img = getTrainTransforms(222)
    print(img["image"])
    

    
    