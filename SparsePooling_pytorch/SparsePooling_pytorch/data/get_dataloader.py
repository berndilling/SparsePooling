import torch
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
from torchvision.transforms import transforms
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from IPython import embed

from SparsePooling_pytorch.utils import utils

# from https://github.com/lpjiang97/sparse-coding/blob/master/src/model/ImageDataset.py
class NatPatchDataset(Dataset):
    def __init__(self, opt, N_per_image:int, width:int, height:int, border:int=0, fpath:str='./datasets/Olshausen/IMAGES.mat'):
        super(NatPatchDataset, self).__init__()
        self.N_per_image = N_per_image
        self.width = width
        self.height = height
        self.border = border
        if opt.dataset_type=="moving":
            self.border = opt.sequence_length
        self.fpath = fpath
        self.images = None
        self.preprocess = opt.preprocess
        self.sequence_length = opt.sequence_length
        # initialize patches
        self.extract_patches_(opt)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]

    def load_data_(self):
        X = loadmat(self.fpath)
        X = X['IMAGES']
        #X = X['IMAGESr']
        img_size = X.shape[0]
        n_img = X.shape[2]
        return X, img_size, n_img

    def preprocess_patches_(self, images):
        s = images.shape
        # preprocessing of image patches
        images = images.reshape(s[0], self.width * self.height) # n_patches, w * h
        if self.preprocess == "subtractmean":
            images = utils.subtractmean(images) # subtract pixel-wise average
        elif self.preprocess=="whiten":
            images = utils.whiten(images) #ZCA whitening of the extracted patches
        return images.reshape(s) # # N_per_image * n_imgs, w, h

    # convention: append temporal dimension to batch dimension -> parallel processing!
    # for SC: either use only one instance of sequence or whole seq
    # for SFA: unrol in time and do SFA stuff
    def create_sequence_(self, img, x, y):
        seq = torch.zeros((self.sequence_length, self.width, self.height))
        dir = np.random.randint(-1,2,(2,))
        while sum(abs(dir))==0:
            dir = np.random.randint(-1,2,(2,)) # sample random direction != (0, 0)
        for t in range(self.sequence_length):
            x_shift = x + t * dir[0]
            y_shift = y + t * dir[1]
            seq[t, :, :] = img[x_shift:x_shift+self.width, y_shift:y_shift+self.height]
        return seq

    def extract_patches_(self, opt):
        X, img_size, n_img = self.load_data_()
        n_patches = self.N_per_image * n_img
        if opt.dataset_type=="moving":
            n_patches *= self.sequence_length

        images = torch.zeros((n_patches, self.width, self.height)) # n_patches, w, h
        # for every image
        counter = 0
        for i in range(n_img):
            img = torch.tensor(X[:, :, i])
            for j in range(self.N_per_image):
                x = np.random.randint(self.border, img_size - self.width - self.border)
                y = np.random.randint(self.border, img_size - self.height - self.border)
                if opt.dataset_type=="moving":
                    seq = self.create_sequence_(img, x, y)
                    images[counter*self.sequence_length:(counter+1)*self.sequence_length, :, :] = seq
                else:
                    crop = img[x:x+self.width, y:y+self.height]
                    images[counter, :, :] = crop
                counter += 1

        images = self.preprocess_patches_(images)
        self.images = images.unsqueeze(1) # N_per_image * n_img, 1, w, h as expected by conv layers

def get_dataloader(opt):
    if opt.dataset=="olshausen":
        dataset = NatPatchDataset(opt, opt.N_patches_per_image, opt.patch_size, opt.patch_size)
        if opt.dataset_type=="moving":
            opt.batch_size_multiGPU *= dataset.sequence_length
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size_multiGPU, shuffle=True, num_workers=16)
    else:
        raise ValueError("dataset not implemented yet!")

    return data_loader
