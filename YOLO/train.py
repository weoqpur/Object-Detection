import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from utils import init_weights
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import Detect
from data_loader import Dataset, RandomCrop, Resize, Normalization
from loss import YoloLoss
from utils import load, save, get_bboxes, mean_average_precision

parser = argparse.ArgumentParser(description='Train Pix2Pix')
parser.add_argument('--num_epochs', default=20, type=int, help='training epoch number')
parser.add_argument('--batch_size', default=16, type=int, help='map data batch size')
parser.add_argument('--load_model', default=False, type=bool)
parser.add_argument('--ckpt_dir', default='./epochs', type=str)

opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = opt.num_epochs

transform_train = transforms.Compose([
    Resize(shape=(448, 448, 3)),
])

# data loading
train_set = Dataset('./maps/train', transform=transform_train)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)

num_batch_train = int((train_set.__len__() / opt.batch_size) + ((train_set.__len__() / opt.batch_size) != 0))

out_path = 'training_results/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

yolo = Detect().to(device)

init_weights(Detect)

fn_loss = YoloLoss()

optimizer = optim.Adam(yolo.parameters(), lr=0.00002, weight_decay=0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)

fn_tonumpy = lambda x: x.to('cpu').datach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean

st_epoch = 0
if opt.load_model:
    yolo, optimizer, st_epoch = load(ckpt_dir=opt.ckpt_dir, model=yolo, optim=optimizer)

for epoch in range(st_epoch, num_epochs):
    loop = tqdm(train_loader, leave=True)
    yolo.train()
    mean_loss = list()

    for batch, data in enumerate(loop):
        image = data['image'].to(device)
        label = data['label'].to(device)

        output = yolo(image)

        loss = fn_loss(output, label)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())

        print('train epoch %04d/%04d | batch %04d/%04d | loss %.4f |' % (num_epochs, epoch, num_batch_train, batch,
                                                                         np.mean(mean_loss)))

    pred_boxes, target_boxes = get_bboxes(
        train_loader, yolo, iou_threshold=0.5, threshold=0.4
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(f"Train mAP: {mean_avg_prec}")

    scheduler.step(mean_avg_prec)

    save(opt.ckpt_dir, yolo, optimizer, epoch)
