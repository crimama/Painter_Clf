import os 
import warnings 
warnings.filterwarnings('ignore')
import cv2 
import pandas as pd 
from glob import glob 
from tqdm import tqdm 
import random 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from easydict import EasyDict as edict
import json 
import cv2 
import sys 
sys.path.append('../../utils')

import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
import torchvision.models as models 

from utils import img_dset
from utils.Dataset import Data_loader,train_valid_split,Data_init
from utils.Augmentation import valid_augmenter,train_augmenter,dacon_augmenter
from utils.Trainer import epoch_run,valid_epoch_run
from utils import Model
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True   

def model_save(model,path,model_name):
    torch.save(model,path+f'save_models/{model_name}')    

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score
def Save_record(record):
    save_record = pd.DataFrame(record)
    save_record.columns = ['fold','epoch_loss','acc','epoch_val_loss','val_acc']
    save_record.to_csv(f'{cfg.save_path}/record.csv',index=False)

if __name__ == "__main__":

    cfg = edict({
        'img_size' : 256,
        'batch_size' :4,
        'train_ratio' : 0.8, 
        'num_epochs' : 30, 
        'num_fold' : 1, 
        'model_name' : 'efficientnet_b0',
        'lr' : 1e-4, 
        'device' : 'cuda:0',
        'save_path' : './Save_models/cait_s36/',
        'seed' : 41 ,
        'crop_ratio' : 0.5 
    })
    seed_everything(cfg.seed) # Seed 고정

    record = [] 
    img_dirs,labels = Data_init('./Data')
    for k,(train_index,valid_index) in enumerate(StratifiedKFold(n_splits=5).split(img_dirs,labels)):        
        #데이터 로드 
        if k == 0:
            label_encoder = {value:key for key,value in enumerate(np.unique(labels))}
        else:
            img_dirs, labels = Data_init('./Data')
        labels = pd.Series(labels).apply(lambda x : label_encoder[x]).values

        #Train - Valid split 
        train_img_dirs, valid_img_dirs = img_dirs[train_index],img_dirs[valid_index]
        train_labels, valid_labels = labels[train_index],labels[valid_index]
        
        #데이터셋, 데이터 로더 
        train_loader = Data_loader(train_img_dirs,train_labels,cfg,augmenter=dacon_augmenter(cfg))
        valid_loader = Data_loader(valid_img_dirs,valid_labels,cfg,augmenter=valid_augmenter(cfg),shuffle=False)
        
        #모델 config 
        model = Model(cfg.model_name,num_classes=len(np.unique(labels))).to(cfg.device)
        criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
        optimizer = torch.optim.RAdam(model.parameters(),lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0)
        scaler = torch.cuda.amp.GradScaler()
        
        print(f'Current Fold : {k}')
        best = 0
        for epoch in tqdm(range(cfg.num_epochs)):
            epoch_loss,acc = epoch_run(train_loader,model,cfg.device,optimizer,criterion,scheduler,scaler)
            epoch_loss_valid, acc_valid = valid_epoch_run(valid_loader,model,cfg.device,criterion)
            print('\nEpoch ', epoch)
            print(f'Train Loss : {epoch_loss} | Train F1 : {acc}')
            print(f'Valid Loss : {epoch_loss_valid} | Valid F1 : {acc_valid}')
            record.append([k,epoch_loss,acc,epoch_loss_valid,acc_valid])
            if epoch == 0:
                best = acc_valid 
                if os.path.exists(cfg.save_path) == False:
                    os.mkdir(cfg.save_path)
                else:
                    torch.save(model,cfg.save_path + f'{k}_fold_best.pt')
            else:
                if acc_valid > best:
                    best = acc_valid 
                    torch.save(model,cfg.save_path + f'{k}_fold_best.pt')
                    print(f'Best_save{epoch}')

        save_record = pd.DataFrame(record)
        save_record.columns = ['fold','epoch_loss','Acc','Valid_epoch_loss','Valid_acc']
        save_record.to_csv(f'{cfg.save_path}record.csv')

    df = pd.DataFrame(cfg.values()).T
    df.columns = cfg.keys() 
    df.to_csv(f'{cfg.save_path}cfg.csv')

