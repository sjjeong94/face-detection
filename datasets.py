import cv2
import copy
import numpy as np
import torch
import torchvision


class WIDERFace(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.WIDERFace(root, split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.asarray(image)
        label = copy.deepcopy(label)
        label['bbox'] = label['bbox'].float().numpy()
        if self.transform:
            image, label = self.transform(image, label)
        return image, label
