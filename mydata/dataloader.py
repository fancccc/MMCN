# -*- coding: utf-8 -*-
'''
@file: dataloader.py
@author: fanc
@time: 2024/5/6 21:16
'''

import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import json
from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms

class DentexDataset(Dataset):
    def __init__(self, root, resize=(640, 640)):
        self.root = root
        self.infos = json.load(open(os.path.join(self.root, 'new_infos.json'), 'r'))
        self.categories = len([x for x in self.infos[0]['annotations'][0] if 'category_id' in x])
        print('categories: {}'.format(self.categories))
        self.resize = resize
        self.transform = transforms.Resize(resize)


    def __getitem__(self, index):
        info = self.infos[index]
        filename = info['file_name']
        img = Image.open(os.path.join(self.root, 'xrays', filename)).convert('RGB')
        bboxes = [ann['bbox'] for ann in info['annotations']]
        labels = [[i[f'category_id_{c}'] for c in range(1, self.categories+1)] for i in info['annotations']]

        img = self.transform(img)

        scale_w = self.resize[1] / info['width']
        scale_h = self.resize[0] / info['height']
        for i in range(len(bboxes)):
            bboxes[i][0] *= scale_w / self.resize[1]
            bboxes[i][2] *= scale_w / self.resize[1]
            bboxes[i][1] *= scale_h / self.resize[0]
            bboxes[i][3] *= scale_h / self.resize[0]

            bboxes[i][0] += bboxes[i][2] / 2
            bboxes[i][1] += bboxes[i][3] / 2
            ## cxcywh

            # bboxes[i][2] = bboxes[i][0] + bboxes[i][2]
            # bboxes[i][3] = bboxes[i][1] + bboxes[i][3]
            ## xyxy

            # bboxes[i][0] *= scale_w
            # bboxes[i][2] *= scale_w
            # bboxes[i][1] *= scale_h
            # bboxes[i][3] *= scale_h
        # img, bboxes = self.cropping(img, bboxes)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        img = transforms.functional.to_tensor(img)
        nl = bboxes.shape[0]
        batch = {'img': img, 'bboxes': bboxes, 'cls': labels, 'ratio_pad': (scale_h, scale_w),
                 'batch_idx' : torch.zeros(nl)}

        return batch

    def __len__(self):
        return len(self.infos)

class PatchDataset(Dataset):
    def __init__(self, img_dir, text_dir):
        super(PatchDataset, self).__init__()
        self.img_dir = img_dir
        self.text_dir = text_dir
        self.imgs = list(filter(lambda x: 'png' in x, os.listdir(img_dir)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img_path = os.path.join(self.img_dir, self.imgs[i])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256))
        img = transforms.functional.to_tensor(img)

        text_path = os.path.join(self.text_dir, self.imgs[i].replace('png', 'txt'))
        with open(text_path, 'r') as f:
            text = json.load(f)

        label = int(self.imgs[i].split('_')[-1].replace('.png', ''))

        # 针对两个不同的类别数量进行独热编码
        num_classes_1 = 8  # 索引-1的类别数
        num_classes_2 = 4  # 索引-2的类别数
        text_tensor_1 = torch.zeros(num_classes_1)
        text_tensor_2 = torch.zeros(num_classes_2)

        # 独热编码设置
        text_tensor_1[int(text[-1])] = 1  # 第一个索引的类别
        text_tensor_2[int(text[-2])] = 1  # 第二个索引的类别

        # 合并两个独热编码向量为一个tensor
        text_tensor = torch.cat((torch.tensor(text[:-2]),text_tensor_1, text_tensor_2), dim=0).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.int64)

        return img, text_tensor, label





if __name__ == '__main__':
    # root = '/home/zcd/datasets/DENTEX/training_data/quadrant'
    # dataset = DentexDataset(root)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8)
    # for batch in dataloader:
    #     print(batch.keys())
    #     break
    img_dir = '/home/zcd/datasets/DENTEX/training_data/quadrant_enumeration_disease/cropped'
    text_dir = '/home/zcd/datasets/DENTEX/training_data/quadrant_enumeration_disease/texts'
    dataset = PatchDataset(img_dir, text_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    for img, text, label in dataloader:
        print(img.shape, text.shape, label.shape, text)
        break