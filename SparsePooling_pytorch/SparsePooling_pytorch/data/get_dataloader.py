import torch
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
import numpy as np
from torchvision.transforms import transforms
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

# from https://github.com/lpjiang97/sparse-coding/blob/master/src/model/ImageDataset.py
class NatPatchDataset(Dataset):

    def __init__(self, N_per_image:int, width:int, height:int, border:int=4, fpath:str='./datasets/Olshausen_whitened/IMAGES.mat'):
        super(NatPatchDataset, self).__init__()
        self.N_per_image = N_per_image
        self.width = width
        self.height = height
        self.border = border
        self.fpath = fpath
        self.images = None
        # initialize patches
        self.extract_patches_()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]

    def extract_patches_(self):
        # load mat
        X = loadmat(self.fpath)
        X = X['IMAGES']
        img_size = X.shape[0]
        n_img = X.shape[2]
        images = torch.zeros((self.N_per_image * n_img, self.width, self.height)) # N_per_image, w, h
        # for every image
        counter = 0
        for i in range(n_img):
            img = X[:, :, i]
            for j in range(self.N_per_image):
                x = np.random.randint(self.border, img_size - self.width - self.border)
                y = np.random.randint(self.border, img_size - self.height - self.border)
                crop = torch.tensor(img[x:x+self.width, y:y+self.height])
                images[counter, :, :] = crop - crop.mean()
                counter += 1
        self.images = images.unsqueeze(1) # N_per_image, 1, w, h as expected by conv layers


def get_dataloader(opt):
    train_dataset = NatPatchDataset(1000, 10, 10)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=True, num_workers=16
    )
    return train_loader

