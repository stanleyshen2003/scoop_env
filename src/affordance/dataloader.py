import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData
import random
import json
import cv2
import numpy as np
from torchvision import transforms

class Dataset_affordance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, mode='train'):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        random.seed(0)
        # data/tool/traj_index/label/*.png
        image_files = glob('preprocessed_data/*/*/*_rgb/*')
        
        
        # labels = [ take tool / scoop / fork / cut / None ]
        
        self.new_image_files = []
            
        # dataset imbalance
        none_ratio = 0.25
        for file in image_files:
            temp = file.split('/')
            if temp[-2][0] == '0':
                if int(temp[-1][:3]) > 30:
                    if random.random() < none_ratio:
                        continue
            self.new_image_files.append(file)
        random.shuffle(self.new_image_files)

        
        if mode == 'train':
            self.new_image_files = self.new_image_files[:int(len(image_files)*0.5)]

        elif mode == 'val':
            self.new_image_files = self.new_image_files[int(len(image_files)*0.5):]
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop((crop_h, crop_w)),
            # transforms.Resize((img_h, img_w))
        ])
        
        print('init_success')
            
    def __len__(self):
        return len(self.new_image_files)

    def __getitem__(self, index):
        file = self.new_image_files[index]
        
        temp = file.split('/')
        if temp[-2][0] == '0':
            if int(temp[-1][:3]) < 30:
                label = 0
            else:
                label = 4
        elif temp[-2][0] == '1':
            if temp[1] == 'spoon':
                label = 1
            elif temp[1] == 'fork':
                label = 2
            elif temp[1] == 'knife':
                label = 3
                    
        # read the png file
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        depth_file = file.replace('_rgb', '_depth')
        depth = np.array(cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE), dtype=np.float32) / 255.0
        depth = np.expand_dims(depth, axis=0)
        return torch.cat((torch.tensor(img), torch.tensor(depth)), dim = 0), label


