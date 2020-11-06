#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implement according to: https://github.com/bobbens/sketch_simplification
# Paper: https://arxiv.org/pdf/1703.08966.pdf

# pytorch 0.4.1

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.serialization import load_lua
from PIL import Image

conv_ske_dic = './Data/convertedSketch/'
simp_ske_dic = './Data/simplifiedSketch/'

cache  = load_lua('model_gan.t7')
model  = cache.model
immean = cache.mean
imstd  = cache.std
model.evaluate()
use_cuda = torch.cuda.device_count() > 0


class AnimeSketchDataset(Dataset):
    """Anime Sketches dataset converted from the original pictures"""
    
    def __init__(self, img_names, root_dir, transform=None):
        
        """
        Args:
            img_names (string): a path to a txt file of sketches file names seperated by comma.
            root_dir (string): directory with all converted sketches.
            transform (callable, optional): optional transform to be applied on a sample.
        """
        
        self.names = open(img_names, 'r').read().split(',')
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        
        img = os.path.join(self.root_dir, self.names[idx])
        data = Image.open(img).convert('L')
        
        if self.transform:
            data = self.transform(data)
            
        return {'image': data, 'image_name': self.names[idx]}
        
class Preprocess():
    """Preprocess the images as proposed in the paper"""
    
    def __init__(self, immean, imstd):
        
        self.mean = immean
        self.std = imstd
    
    def __call__(self, data):
        
        w, h = data.size[0], data.size[1]
        pw = 8 - (w%8) if w%8 != 0 else 0
        ph = 8 - (h%8) if h%8 != 0 else 0
        
        data = (transforms.ToTensor()(data) - immean) / imstd
        
        if pw!=0 or ph!=0:
            data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
            
        return data


def simplify(img_names, root_dir, dest_dir, immean, imstd, transform, batch_size, shuffle=False):

	dataset = AnimeSketchDataset(img_names, root_dir, Preprocess(immean, imstd))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    for inputs in dataloader:
        data, img = inputs['image'], inputs['image_name']
        if use_cuda:
            pred = model.cuda().forward(inputs.cuda()).float()
        else:
            pred = model.forward(data)
        for i in range(data.shape[0]):
            save_image(pred[i], dest_dir + img[i])

