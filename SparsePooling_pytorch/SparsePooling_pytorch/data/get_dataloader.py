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
        raise Exception("dataset not implemented yet!")

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

# from https://github.com/loeweX/Greedy_InfoMax
def get_stl10_dataloader(opt):
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
        [get_transforms(eval=True, aug=aug["stl10"])]
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


def get_transforms(eval=False, aug=None):
    trans = []

    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"]))

    if aug["randcrop"] and eval:
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
