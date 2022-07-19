import os
import time
import logging
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import models
import datasets
import transforms


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class IOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, gt):
        o_l, o_t, o_r, o_b = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        g_l, g_t, g_r, g_b = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]

        filt = g_l > 0
        o_l, o_t, o_r, o_b = o_l[filt], o_t[filt], o_r[filt], o_b[filt]
        g_l, g_t, g_r, g_b = g_l[filt], g_t[filt], g_r[filt], g_b[filt]

        o_a = (o_t + o_b) * (o_l + o_r)
        g_a = (g_t + g_b) * (g_l + g_r)

        iw = torch.minimum(o_l, g_l) + torch.minimum(o_r, g_r)
        ih = torch.minimum(o_t, g_t) + torch.minimum(o_b, g_b)
        inter = iw * ih
        union = o_a + g_a - inter
        iou = inter / (union + 1e-9)
        loss = -torch.log(iou)
        return loss.mean()


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = IOULoss()

    def forward(self, out, gt):
        # centerness
        gt_c = gt[:, 4]
        out_c = out[:, 4]
        loss_c = self.bce(out_c, gt_c)

        gt_r = gt[:, :4]
        out_r = out[:, :4]
        loss_r = self.iou(out_r, gt_r)

        loss = loss_c + loss_r

        self.loss_c = loss_c.detach()
        self.loss_r = loss_r.detach()

        return loss


def train(
    logs_root,
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    batch_size=8,
    epochs=5,
    num_workers=2,
):

    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(
        os.path.join(logs_root, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = models.Detector()
    net = net.to(device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']

    T_train = transforms.Compose([
        transforms.RandomResize(512, 640),
        transforms.RandomCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    T_val = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.WIDERFace('./data', 'train', T_train)
    val_dataset = datasets.WIDERFace('./data', 'val', T_val)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    criterion = Loss()

    # logger.info(net)
    # logger.info(device)
    logger.info(optimizer)

    logger.info('| %5s | %8s | %8s | %8s | %8s | %8s |' %
                ('epoch', 'time', 'loss T', 'loss V', 'loss_c', 'loss_r'))

    for epoch in range(epoch_begin, epochs):
        t0 = time.time()
        net.train()
        losses = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = net(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            losses += loss.detach()

        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        t0 = time.time()
        net.eval()
        losses = 0
        losses_c = 0
        losses_r = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)

                out = net(x)
                loss = criterion(out, y)

                losses += loss.detach()

                losses_c += criterion.loss_c
                losses_r += criterion.loss_r

        loss_val = losses / len(val_loader)
        loss_k = losses_c / len(val_loader)
        loss_o = losses_r / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total = time_train + time_val

        logger.info('| %5d | %8.1f | %8.4f | %8.4f | %8.4f | %8.4f |' %
                    (epoch + 1, time_total, loss_train, loss_val, loss_k, loss_o))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)


if __name__ == '__main__':
    train(
        logs_root='logs/widerface/fcos2',
        learning_rate=0.001,
        epochs=50,
    )
