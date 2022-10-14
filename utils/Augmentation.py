import torchvision.transforms as transforms 
import torch 
import torch.nn as nn 
import numpy as np 
class RandomCrop(nn.Module):
    def __init__(self,ratio):
        super(RandomCrop,self).__init__()
        self.ratio = ratio 

    def forward(self,img):
        height = img.shape[0]
        weight = img.shape[1]

        after_height = int(height*self.ratio)
        after_weight = int(weight*self.ratio)

        start_weight = np.random.choice(weight-after_weight)
        start_height = np.random.choice(height-after_height)

        after_img = img[start_weight:start_weight+after_weight,
                        start_height:start_height+after_height]
        return after_img 


def valid_augmenter(cfg):
    augmenter = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(cfg.img_size),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    return augmenter 

def train_augmenter(cfg):
    augmenter = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size = (cfg.img_size,cfg.img_size)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96,scale=(0.5, 1.0)),
            transforms.ColorJitter(brightness=0.2)
        ])
    return augmenter 

def dacon_augmenter(cfg):
    train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(cfg.img_size),
        transforms.Resize(size = (cfg.img_size,cfg.img_size)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])    
    return train