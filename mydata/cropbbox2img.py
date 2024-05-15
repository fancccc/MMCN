# -*- coding: utf-8 -*-
'''
@file: cropbbox2img.py
@author: fanc
@time: 2024/5/10 16:35
'''
import os
import numpy as np
import json

path = '/home/zcd/datasets/DENTEX/training_data/quadrant_enumeration_disease/'
with open(os.path.join(path, 'new_infos.json'), 'r')as f:
    coco = json.load(f)
with open(os.path.join(path, 'q_anns.json'), 'r')as f:
    q_ann = json.load(f)
with open(os.path.join(path, 'e_anns.json'), 'r')as f:
    e_ann = json.load(f)