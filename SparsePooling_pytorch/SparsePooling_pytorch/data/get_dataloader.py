from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from IPython import embed

from SparsePooling_pytorch.utils import utils

# from https://github.com/lpjiang97/sparse-coding/blob/master/src/model/ImageDataset.py
class NatPatchDataset(Dataset):
    def __init__(self, opt, n_patches:int, width:int, height:int, border:int=0, fpath:str='./datasets/Olshausen/IMAGES.mat', transform=None):
        super(NatPatchDataset, self).__init__()
        self.n_patches = n_patches
        self.width = width
        self.height = height
        self.border = border
        if opt.dataset_type=="moving":
            self.border = opt.sequence_length
        self.fpath = fpath
        self.images = None
        self.preprocess = opt.preprocess
        self.sequence_length = opt.sequence_length
        self.n_channels = None # will be defined in extract_patches_
        self.transform = transform # used for stl10, overwritten in load_data_()
        self.extract_patches_(opt) # initialize patches

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_data_(self, opt):
        if opt.dataset == "olshausen":
            X = loadmat(self.fpath)
            X = torch.tensor(X['IMAGES']).unsqueeze(-2) # 512, 512, 1, 10
            #X = X['IMAGESr']
            self.n_channels = 1
        elif opt.dataset == "stl10":
            _, _, _, transform_valid, unsupervised_dataset, _, _ = get_stl10(opt, ccrop=False) # 100000, 3, 96, 96
            X = torch.tensor(unsupervised_dataset.data).permute(2, 3, 1, 0) # 96, 96, 3, 100000 
            self.n_channels = 3
            
            #transform_valid.transforms.insert(0, transforms.ToPILImage())
            #self.transform = transform_valid
            
            self.transform = transforms.Compose([transforms.ToPILImage(), 
                                                    transforms.Grayscale(), 
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=0.562, std=0.245)])
            # numbers for mean and std obtained by printing mean and std of returns of train_loader for large batch_size (wihtout Normalize)
        else:
            raise Exception("dataset not implemented yet!")

        img_size = X.shape[0]
        n_img = X.shape[-1]
        return X.permute(3, 2, 0, 1), img_size, n_img # X: n_img, n_channels, img_size, img_size

    def preprocess_patches_(self, images): # n_patches, n_channels, w, h
        s = images.shape
        images = images.reshape(s[0], self.n_channels * self.width * self.height) # n_patches, w * h
        if self.preprocess == "subtractmean":
            images = utils.subtractmean(images) # subtract pixel-wise average
        elif self.preprocess=="whiten":
            images = utils.whiten(images) # ZCA whitening of the extracted patches
        return images.reshape(s) # n_patches, n_channels, w, h

    # convention: append temporal dimension to batch dimension -> parallel processing!
    # for SC: either use only one instance of sequence or whole seq
    # for SFA: unrol in time and do SFA stuff
    def create_sequence_(self, img, x, y): # img: n_channels, img_size, img_size
        seq = torch.zeros((self.sequence_length, self.n_channels, self.width, self.height))
        dir = np.random.randint(-1,2,(2,))
        while sum(abs(dir))==0:
            dir = np.random.randint(-1,2,(2,)) # sample random direction != (0, 0)
        for t in range(self.sequence_length):
            x_shift = x + t * dir[0]
            y_shift = y + t * dir[1]
            seq[t, :, :, :] = img[:, x_shift:x_shift+self.width, y_shift:y_shift+self.height]
        return seq

    def extract_patches_(self, opt):
        X, img_size, n_img = self.load_data_(opt) # X: n_img, n_channels, img_size, img_size
        
        n_per_image = self.n_patches // n_img
        if n_per_image == 0:
            print("ATTENTION: the dataset contains more images than image patches to extract!"
                " -> n_per_image is raised to 1, which leads to n_patches = ", n_img, " in total")
            n_per_image = 1
            self.n_patches = n_per_image * n_img
        if opt.dataset_type=="moving":
            self.n_patches *= self.sequence_length

        images = torch.zeros((self.n_patches, self.n_channels, self.width, self.height)) # n_patches, n_channels, w, h
        # for every image
        counter = 0
        for i in range(n_img):
            img = X[i, :, :, :] # n_channels, img_size, img_size
            for j in range(n_per_image):
                x = np.random.randint(self.border, img_size - self.width - self.border)
                y = np.random.randint(self.border, img_size - self.height - self.border)
                if opt.dataset_type=="moving":
                    seq = self.create_sequence_(img, x, y)
                    images[counter*self.sequence_length:(counter+1)*self.sequence_length, :, :, :] = seq
                else:
                    crop = img[:, x:x+self.width, y:y+self.height]
                    images[counter, :, :, :] = crop
                counter += 1

        # for STL-10 preprocessing will be done by dataloader (see below)
        if opt.dataset == "olshausen":
            images = self.preprocess_patches_(images)
        
        self.images = images # n_patches (= n_per_image * n_img), n_channels, w, h as expected by conv layers


class BarsDataset(Dataset):
    def __init__(self, opt, size = 10):
        super(BarsDataset, self).__init__()

        print("Using Dataset of moving bars with size: ", size, ". That means n_patches = 2 * size = ", 2*size, ".")
        print("Using old n_patches as n_sequences: ", opt.n_patches, " using sequence_length: ", opt.sequence_length)
        self.sequence_length = opt.sequence_length
        self.n_sequences = opt.n_patches
        
        # Create patches
        X = torch.zeros(2 * size, 1, size, size) # n_img, n_channels, img_size, img_size
        for i in range(size):
            X[i, 0, i, :] = 1. # horizontal bars
            X[i+size, 0, :, i] = 1. # vertical bars
        
        if opt.dataset_type=="moving":
            # TODO implement sequences!
            X_seq = torch.zeros(self.n_sequences * self.sequence_length, 1, size, size)
            for i in range(self.n_sequences):
                # select if hor. (0:size-1) or vert. (size:2*size)
                orientation = torch.randint(0, 2, (1,)) * size
                for t in range(self.sequence_length):
                    # select instance of bar
                    inst = torch.randint(0, size, (1,))
                    X_seq[i*self.sequence_length + t, :, :, :] = X[orientation + inst, :, :, :]
            
            self.images = X_seq
        else:
            raise Exception("BarsDataset should be used with dataset_type = 'moving'!")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        return img
        

#############################################################################################################################
# main functions

def get_dataloader(opt):
    # TODO finalise that:
    if opt.dataset == "bars":
        dataset = BarsDataset(opt)
    else:
        dataset = NatPatchDataset(opt, opt.n_patches, opt.patch_size, opt.patch_size)
    
    if opt.dataset_type=="moving":
        opt.batch_size_multiGPU *= dataset.sequence_length
    
    # shuffle = False important here if "moving" dataset is used because temporal and batch dimension are merged! 
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16)

    return data_loader

def get_dataloader_class(opt):
    if opt.dataset_class=="stl10":
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_stl10_dataloader(
            opt
        )
    else:
        raise Exception("dataset not implemented yet!")
    
    return (
        train_loader,
        train_dataset,
        supervised_loader,
        supervised_dataset,
        test_loader,
        test_dataset,
    )

#################################################################################################
# STL-10 import taken and adapted from https://github.com/loeweX/Greedy_InfoMax

def get_stl10(opt, ccrop=True):
    base_folder = os.path.join(opt.data_input_dir, "stl10_binary")

    aug = {
        "stl10": {
            "randcrop": opt.random_crop_size,
            "flip": True,
            "grayscale": opt.grayscale,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }
    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["stl10"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["stl10"], ccrop=ccrop)]
    )

    unsupervised_dataset = torchvision.datasets.STL10(
        base_folder,
        split="unlabeled",
        transform=transform_train,
        download=opt.download_dataset,
    ) #set download to True to get the dataset

    train_dataset = torchvision.datasets.STL10(
        base_folder, split="train", transform=transform_train, download=opt.download_dataset
    )

    test_dataset = torchvision.datasets.STL10(
        base_folder, split="test", transform=transform_valid, download=opt.download_dataset
    )

    return base_folder, aug, transform_train, transform_valid, unsupervised_dataset, train_dataset, test_dataset 


def get_stl10_dataloader(opt):
    base_folder, aug, transform_train, transform_valid, unsupervised_dataset, train_dataset, test_dataset = get_stl10(opt)

    # default dataset loaders, do not shuffle if you create a dataset for hidden reps for classification
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle= not opt.create_hidden_representation, num_workers=16
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=16,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )

    # create train/val split
    if opt.validate:
        print("Use train / val split")

        if opt.training_dataset == "train":
            dataset_size = len(train_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        elif opt.training_dataset == "unlabeled":
            dataset_size = len(unsupervised_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            unsupervised_loader = torch.utils.data.DataLoader(
                unsupervised_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        else:
            raise Exception("Invalid option")

        # overwrite test_dataset and _loader with validation set
        test_dataset = torchvision.datasets.STL10(
            base_folder,
            split=opt.training_dataset,
            transform=transform_valid,
            download=opt.download_dataset,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size_multiGPU,
            sampler=valid_sampler,
            num_workers=16,
        )

    else:
        print("Use (train+val) / test split")

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


def create_validation_sampler(dataset_size):
    # Creating data indices for training and validation splits:
    validation_split = 0.2
    shuffle_dataset = True

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def get_transforms(eval=False, aug=None, ccrop=True):
    trans = []

    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"]))

    if aug["randcrop"] and eval and ccrop:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans
