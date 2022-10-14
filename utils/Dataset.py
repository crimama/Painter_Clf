import numpy as np 
import torch 
from torch.utils.data import Dataset,DataLoader
import cv2 
import pandas as pd 
from glob import glob 

class img_dset(Dataset):
    def __init__(self,img_dirs,labels,augmenter,cfg):
        super(img_dset,self).__init__()
        self.img_dirs = img_dirs 
        self.labels = labels 
        self.augmenter = augmenter 
        self.cfg = cfg 
    def __len__(self):
        return len(self.img_dirs)
        
    def img_load(self,img_dir):
        return cv2.imread(img_dir)

    def __getitem__(self,idx):
        #img load 
        img_dir = self.img_dirs[idx]
        img = self.img_load(img_dir)
        #resize 
        img = self.augmenter(img)

        #label load 
        label = self.labels[idx]
        return img, label 


def Data_loader(img_dirs,labels,cfg,augmenter,shuffle=True):
    Data_set = img_dset(img_dirs,labels,augmenter,cfg)
    Data_loader = DataLoader(Data_set,batch_size=cfg.batch_size,shuffle=shuffle)
    return Data_loader


def train_valid_split(data,ratio):
    indx = int(len(data)*ratio)
    train_data = data[:indx]
    valid_data = data[indx:]
    return train_data, valid_data 

def Data_init(path,shuffle=False):
    df = pd.read_csv(f'{path}/train.csv')
    df['img_path'] = df['img_path'].apply(lambda x:'./Data/'+ x.split('./')[1])

    img_dirs = df['img_path'].values
    labels = df['artist'].values
    
    if shuffle:
        shuffle_indx = np.arange(len(img_dirs))
        np.random.shuffle(shuffle_indx)

        img_dirs = img_dirs[shuffle_indx]
        labels = labels[shuffle_indx]
    return img_dirs,labels 
