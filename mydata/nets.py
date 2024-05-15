# -*- coding: utf-8 -*-
'''
@file: nets.py
@author: fanc
@time: 2024/5/10 14:31
'''
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, resnet34, resnet101
import math
import torch.nn.functional as F

class PatchNet(nn.Module):
    def __init__(self, num_classes=4, text_len=17, embedding_dim=256, num_heads=8, device='cuda'):
        super().__init__()
        resnet = resnet50(weights=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # self.conv1x1 = nn.Conv2d(2048, 8, kernel_size=1)
        # self.pool = nn.AdaptiveAvgPool2d((8, 1))

        # text
        self.text_fc = nn.Linear(text_len, embedding_dim)
        self.text_encod = TransformerEncoder(embedding_dim, num_heads, 0.1, device=device)

        # mutil modal fuse
        self.mhca = MultiHeadCrossAttention(embedding_dim, num_heads)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes))


    def forward(self, x):
        img = x[0]
        text = x[1]
        x = self.backbone(img)
        x = x.view(x.size(0), 8, 256)
        text_encod = self.text_encod(self.text_fc(text))
        out = self.mhca(x, text_encod)
        out = self.fc(out.view(out.size(0), -1))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=17, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, max_len=8, device='cuda'):
        super(TransformerEncoder, self).__init__()
        # self.pos_encoder = PositionalEncoding(embed_size, max_len, device=device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = self.pos_encoder(src)  # 添加位置编码
        # src = self.dropout(src)
        output = self.transformer_encoder(src)
        return output

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        # 图像到文本的注意力机制
        self.img_to_text_keys = nn.Linear(embed_size, embed_size)
        self.img_to_text_values = nn.Linear(embed_size, embed_size)
        self.text_to_img_keys = nn.Linear(embed_size, embed_size)
        self.text_to_img_values = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, img, text):

        # Splitting the embedding into multiple heads
        bsz = img.shape[0]
        # 图像到文本
        img_to_text_keys = self.img_to_text_keys(img).view(bsz, -1, self.num_heads, self.head_dim)
        img_to_text_values = self.img_to_text_values(img).view(bsz, -1, self.num_heads, self.head_dim)
        text_keys = text.repeat(1, img.shape[1], 1).view(bsz, -1, self.num_heads, self.head_dim)

        # 文本到图像
        text_to_img_keys = self.text_to_img_keys(text).view(bsz, -1, self.num_heads, self.head_dim)
        text_to_img_values = self.text_to_img_values(text).view(bsz, -1, self.num_heads, self.head_dim)
        img_keys = img.view(bsz, -1, self.num_heads, self.head_dim)

        # Attention mechanism (using scaled dot product attention)
        img_attention = torch.einsum("bnqd,bnkd->bnqk", text_keys, img_to_text_keys)  # bsz, num_queries, num_keys
        text_attention = torch.einsum("bnqd,bnkd->bnqk", img_keys, text_to_img_keys)

        img_attention = F.softmax(img_attention / self.head_dim ** 0.5, dim=-1)
        text_attention = F.softmax(text_attention / self.head_dim ** 0.5, dim=-1)

        img_out = torch.einsum("bnql,bnld->bnqd", img_attention, img_to_text_values).reshape(bsz, -1, self.embed_size)
        text_out = torch.einsum("bnql,bnld->bnqd", text_attention, text_to_img_values).reshape(bsz, -1, self.embed_size)

        # Concat and pass through final linear layer
        out = torch.cat((img_out, text_out), dim=1)
        out = self.fc_out(out)

        return out

class mresnet50(nn.Module):
    def __init__(self, num_classes=5):
        super(mresnet50, self).__init__()
        self.backbone = resnet50(weights=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        img = x[0]
        return self.backbone(img)

class mresnet18(nn.Module):
    def __init__(self, num_classes=5):
        super(mresnet18, self).__init__()
        self.backbone = resnet18(weights=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        img = x[0]
        return self.backbone(img)

class mresnet34(nn.Module):
    def __init__(self, num_classes=5):
        super(mresnet34, self).__init__()
        self.backbone = resnet34(weights=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        img = x[0]
        return self.backbone(img)

class mresnet101(nn.Module):
    def __init__(self, num_classes=5):
        super(mresnet101, self).__init__()
        self.backbone = resnet101(weights=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        img = x[0]
        return self.backbone(img)

if __name__ == '__main__':
    net = PatchNet(device='cpu')
    img = torch.randn(2, 3, 256, 256)
    text = torch.tensor([[[0.2752, 0.5183, 0.0693, 0.2256, 0.0156, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]],
        [[0.7023, 0.3847, 0.0557, 0.1340, 0.0075, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000]]])
    print(text.shape)
    net([img, text])




