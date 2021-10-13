import torch
import os
from collections import OrderedDict
import cv2
import numpy as np
import random

class AIHUBPedestrianDataset(torch.utils.data.Dataset):
    def __init__(self, dir_image, imgh, imgw, transform=None, train=True):
        super().__init__()
        self.ids = []
        self.paths = []
        self.id_to_label = OrderedDict()
        self.label_to_path = OrderedDict()

        self.imgh = imgh
        self.imgw = imgw
        self.transform = transform
        self.train = train

        for path in os.listdir(dir_image):
            id = path.split('_')[1]
            self.ids.append(id)
            self.paths.append(os.path.join(dir_image, path))
        
        id_list = np.unique(self.ids)
        for i in range(len(id_list)):
            self.id_to_label[id_list[i]] = i
            self.label_to_path[i] = []

        for path, id in zip(self.paths, self.ids):
            self.label_to_path[self.id_to_label[id]].append(path)

        

    def __getitem__(self, idx):
        if self.train:
            path = random.choice(self.label_to_path[idx])
            label = idx
        else:
            path = self.paths[idx]
            label = self.id_to_label[self.ids[idx]]

        img = cv2.imread(path)
        img = cv2.resize(img, (self.imgw, self.imgh))

        if self.transform and self.train:
            aug = self.transform(image=img)
            return aug['image'], label
        else:
            return img, label

    def __len__(self):
        if self.train:
            return len(self.label_to_path)
        else:
            return len(self.paths)

    def size(self):
        return len(self.paths)

class Market1501PedestrianDataset(torch.utils.data.Dataset):
    def __init__(self, dir_image, imgh, imgw, transform=None, train=True):
        super().__init__()
        self.ids = []
        self.paths = []
        self.id_to_label = OrderedDict()
        self.label_to_path = OrderedDict()

        self.imgh = imgh
        self.imgw = imgw
        self.transform = transform
        self.train = train

        for path in os.listdir(dir_image):
            id = int(path.split('_')[0])
            self.ids.append(id)
            self.paths.append(os.path.join(dir_image, path))
        
        id_list = np.unique(self.ids)
        for i in range(len(id_list)):
            self.id_to_label[id_list[i]] = i
            self.label_to_path[i] = []

        for path, id in zip(self.paths, self.ids):
            self.label_to_path[self.id_to_label[id]].append(path)

        

    def __getitem__(self, idx):
        if self.train:
            path = random.choice(self.label_to_path[idx])
            label = idx
        else:
            path = self.paths[idx]
            label = self.ids[idx]

        img = cv2.imread(path)
        img = cv2.resize(img, (self.imgw, self.imgh))

        if self.transform and self.train:
            aug = self.transform(image=img)
            return aug['image'], label
        else:
            return img, label

    def __len__(self):
        if self.train:
            return len(self.label_to_path)
        else:
            return len(self.paths)

    def size(self):
        return len(self.paths)

if __name__ == '__main__':
    dir_image = 'F:/Data/pedestrain_korean/image/Training/data'
    imgh, imgw = 128, 64
    dataset = AIHUBPedestrianDataset(dir_image, imgh, imgw)

    print(dataset.label_to_path.keys())
    for i in range(10):
        dataset.__getitem__(0)

