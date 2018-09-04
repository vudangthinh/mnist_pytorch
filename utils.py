import numpy as np
import torch
from torchvision import transforms

class ToTensor(object):
    def __call__(self, sample):
        label, image = sample
        # image = image.transpose((2, 0, 1))
        label = long(label)

        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0)
        image = image.float()
        return (label, image)
