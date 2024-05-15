# -*- coding: utf-8 -*-
'''
@file: test.py
@author: fanc
@time: 2024/5/7 8:21
'''
from nn.tasks import DetectionModel
import torch
import torch.nn.functional as F
from cfg import get_cfg, get_save_dir
from mydata.dataloader import DentexDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from utils.ops import non_max_suppression
def collate_fn(batch):
    """Collates data samples into batches."""
    new_batch = {}
    keys = batch.keys()
    # values = list(zip(*[[b[i] for i in range(len(b))] for b in batch.values()]))
    # print(values)
    for i, k in enumerate(keys):
        # value = values[i]
        if k == "img":
            # value = torch.stack(value, 0)
            value = batch[k]
        if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
            value = [batch[k][i] for i in range(len(batch[k]))]
            # print(f'{k}:{value}')
            value = torch.cat(value, 0)
        if k in {'batch_idx'}:
            value = [batch[k][i] for i in range(len(batch[k]))]
        new_batch[k] = value
    new_batch["batch_idx"] = list(new_batch["batch_idx"])
    # print(new_batch["batch_idx"])
    for i in range(len(new_batch["batch_idx"])):
        new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        # print(new_batch["batch_idx"][i])
    new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0).unsqueeze(-1)
    return new_batch
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
root = '/home/zcd/datasets/DENTEX/training_data/quadrant'
dataset = DentexDataset(root)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

train_bar = tqdm(enumerate(dataloader), total=len(dataloader))
for idx, batch in train_bar:
    batch = collate_fn(batch)
    batch['img'] = batch['img'].to(device)
    preds = model(batch['img'])
    preds = non_max_suppression(preds, 0.25, 0.45, classes=4)
    print(preds)
    break
