import cv2
import math
import torch
import random
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        h, w, c = image.shape
        if random.random() < self.p:
            image = np.fliplr(image)
            bbox = target['bbox']
            bbox[:, 0] = w - bbox[:, 0] - bbox[:, 2]
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        h, w, c = image.shape
        ws, hs = self.size[0] / w, self.size[1] / h
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LANCZOS4)
        bbox = target['bbox']
        bbox[:, 0] *= ws
        bbox[:, 1] *= hs
        bbox[:, 2] *= ws
        bbox[:, 3] *= hs
        return image, target


class ToTensor:
    def __init__(self):
        return

    def __call__(self, image, target):
        label = label_encode(image, target)
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32) / 255
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return image, label


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).reshape(-1, 1, 1)
        self.std = torch.FloatTensor(std).reshape(-1, 1, 1)

    def __call__(self, image, target):
        image = (image - self.mean) / self.std
        return image, target


def label_encode(image, target, num_classes=91, R=4):
    H, W, C = image.shape
    h, w = H // R, W // R

    gt_k = np.zeros((num_classes, h, w), dtype=np.float32)
    gt_r = np.zeros((4, h, w), dtype=np.float32)

    for bbox in target['bbox']:
        bx, by, bw, bh = bbox / R
        l, t, r, b = int(bx), int(by), int(bx+bw), int(by+bh)

        for y in range(t, b):
            for x in range(l, r):
                if x >= 0 and x < w and y >= 0 and y < h:
                    l_ = x - l
                    t_ = y - t
                    r_ = r - x
                    b_ = b - y
                    v = np.sqrt(min(l_, r_)/max(l_, r_)
                                * min(t_, b_)/max(t_, b_))
                    if gt_k[0, y, x] < v:
                        gt_k[0, y, x] = v
                        gt_r[0, y, x] = l_
                        gt_r[1, y, x] = t_
                        gt_r[2, y, x] = r_
                        gt_r[3, y, x] = b_

    gt = np.concatenate([gt_r, gt_k], 0)

    return gt


class KeyPointExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, 1, 1)

    def forward(self, x):
        y = self.pool(x)
        x = x - 1.0 * (x < y)
        return x

def label_decode(gt, R=4):
    gt_k = gt[4:]

    kpe = KeyPointExtractor()(torch.from_numpy(gt_k).unsqueeze(0)).numpy().squeeze()

    points = np.where(kpe > 0.25)
    decoded = []
    for x, y in zip(points[1], points[0]):
        l = gt[0, y, x]
        t = gt[1, y, x]
        r = gt[2, y, x]
        b = gt[3, y, x]

        bx = (x - l) * R
        by = (y - t) * R
        bw = (l + r) * R
        bh = (t + b) * R

        bbox = [bx, by, bw, bh]

        decoded.append({
            'bbox': bbox,
            'category_id': 0,
        })
    
    return decoded
    