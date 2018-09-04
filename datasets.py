import torch
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from mnist import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MNIST_Datasets(Dataset):
    def __init__(self, mode, path, transform=None):
        super(MNIST_Datasets, self).__init__()

        self.transform = transform
        self.data = []

        for (label, image) in read(mode, path):
            if self.transform:
                label, image = self.transform((label, image))
            self.data.append((label, image))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)