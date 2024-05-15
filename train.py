# # -*- coding: utf-8 -*-
# '''
# @file: train.py
# @author: fanc
# @time: 2024/5/6 17:52
# '''
# from nn.tasks import DetectionModel
# import torch
# import torch.nn.functional as F
# from cfg import get_cfg, get_save_dir
# from mydata.dataloader import DentexDataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from torch.optim.lr_scheduler import ExponentialLR
# import os
# import time
#
# def collate_fn(batch):
#     """Collates data samples into batches."""
#     new_batch = {}
#     keys = batch.keys()
#     # values = list(zip(*[[b[i] for i in range(len(b))] for b in batch.values()]))
#     # print(values)
#     for i, k in enumerate(keys):
#         # value = values[i]
#         if k == "img":
#             # value = torch.stack(value, 0)
#             value = batch[k]
#         if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
#             value = [batch[k][i] for i in range(len(batch[k]))]
#             # print(f'{k}:{value}')
#             value = torch.cat(value, 0)
#         if k in {'batch_idx'}:
#             value = [batch[k][i] for i in range(len(batch[k]))]
#         new_batch[k] = value
#     new_batch["batch_idx"] = list(new_batch["batch_idx"])
#     # print(new_batch["batch_idx"])
#     for i in range(len(new_batch["batch_idx"])):
#         new_batch["batch_idx"][i] += i  # add target image index for build_targets()
#         # print(new_batch["batch_idx"][i])
#     new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0).unsqueeze(-1)
#     return new_batch
#
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# print(device)
# model = DetectionModel('./cfg/models/v8/yolov8.yaml', nc=4)
# args = get_cfg('./cfg/default.yaml')
# model.args = args
# # model = torch.load('/home/zcd/codes/YOLOv8/ultralytics/results/202405072026/model_epoch_9.pt')
# model = model.to(device)
# model.train()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.99))
# scheduler = ExponentialLR(optimizer, gamma=0.99)
# datetime = time.strftime("%Y%m%d%H%M", time.localtime())
# save_dir = os.path.join(f'./results/{datetime}')
# os.makedirs(save_dir, exist_ok=True)
# root = '/home/zcd/datasets/DENTEX/training_data/quadrant'
# dataset = DentexDataset(root)
# dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
# epochs = 20
# loss = None
# tloss = None
#
# for epoch in range(epochs):
#     train_bar = tqdm(enumerate(dataloader), total=len(dataloader))
#     for idx, batch in train_bar:
#         optimizer.zero_grad()
#         batch = collate_fn(batch)
#         batch['img'] = batch['img'].to(device)
#         # batch = {'img': img.to(device),
#         #          'batch_idx': torch.tensor([idx for _ in range(bboxes.shape[1])]),
#         #          'cls': labels,
#         #          'bboxes': bboxes}
#         # for t in batch:
#         #     if isinstance(batch[t], torch.Tensor):
#         #         print(f'{t}: {batch[t].shape}')
#         # temp = torch.cat((batch['batch_idx'], batch["cls"], batch["bboxes"]), 1)
#         # print(temp.shape)
#         # break
#         loss, loss_items = model(batch)
#         # print(loss, loss_items)
#         loss.backward()
#         optimizer.step()
#         train_bar.set_description(
#             f'Epoch: {epoch + 1}/{epochs}, '
#             f'Loss: {loss:.4f}, '
#             f'loss1: {loss_items[0]:.4f}, '
#             f'loss2: {loss_items[1]:.4f}, '
#             f'loss3: {loss_items[2]:.4f}'
#         )
#     scheduler.step()
#     if epoch % 10 == 0 or epoch == epochs - 1:
#         torch.save(model, os.path.join(save_dir, f'model_epoch_{epoch}.pt'))
#
# # # for l in model(img):
# # #     print(l.shape)
from models.yolo.model import YOLO
#
# Load a model
# can use
model = YOLO("./cfg/models/v8/yolov8.yaml")  # build a new model from scratch
# model = YOLO('./runs/detect/train8/weights/best.pt')
model.train(data="./cfg/datasets/dentex_qed.yaml", epochs=200)
metrics = model.val()
#prediction
# model = YOLO('./runs/detect/train6/weights/best.pt')
# model.predict()
