# -*- coding: utf-8 -*-
'''
@file: inference.py
@author: fanc
@time: 2024/5/7 8:24
'''
from nn.tasks import DetectionModel
import torch
import torch.nn.functional as F
from cfg import get_cfg, get_save_dir
from mydata.dataloader import DentexDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import os
from utils import ops
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)
# model = DetectionModel('./cfg/models/v8/yolov8.yaml', nc=4)
args = get_cfg('./cfg/default.yaml')
args.mode = 'predict'
model = torch.load('/home/zcd/codes/YOLOv8/ultralytics/results/202405071603/model_epoch_90.pt')
# print(model)
model.args = args
model = model.to(device)
# weight_path = './results/model_epoch_560.pth'
# model.load_state_dict(torch.load(weight_path))
model.eval()

img_path = '/home/zcd/datasets/DENTEX/training_data/quadrant/xrays/train_0.png'
img = Image.open(img_path).convert('RGB')
orig_shape = img.size
t = transforms.Resize((640, 640))
img = t(img)
img_tensor = transforms.ToTensor()(img).unsqueeze(0)
img_tensor = img_tensor.to(device)
with torch.no_grad():
    out1, out2 = model(img_tensor)
    # print(type(out1), type(out2))
    # print(out1)
    # for l in out2:
    #     print(l.shape)
    preds = ops.non_max_suppression(
        out1,
        0.1,
        0.9,
        classes=4
    )
    print(preds)