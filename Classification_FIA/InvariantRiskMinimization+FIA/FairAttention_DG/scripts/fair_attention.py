import sys, os

import blobfile as bf
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from torchvision.models import *
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR, StepLR

from sklearn.metrics import *
from fairlearn.metrics import *

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torchvision.ops.misc import MLP, Permute

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional



class Cond_EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads
        
        
        self.ln_c = norm_layer(hidden_dim)
        
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input, attr):
        
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        
        q = self.ln_c(attr)
        
        x, _ = self.self_attention(q, x, x, need_weights=False)
        x = self.dropout(x)
        
        
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
                    
                    
class Cond_Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int, #197
        num_layers: int, #1
        num_heads: int,  #12
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.pos_embedding_c = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        
        
        self.dropout = nn.Dropout(dropout)
        self.dropout_c = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = Cond_EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers) 
        self.ln = norm_layer(hidden_dim)

    def forward(self, input, attr):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        attr = attr + self.pos_embedding_c
        input = self.dropout(input)
        attr = self.dropout_c(attr)
        
        for layer in self.layers:
            input = layer(input, attr)
        
        return self.ln(input)

class Cond_VIT(nn.Module):
    def __init__(self, vit_model, patch_size, hidden_dim, num_heads, num_attrs,
                 seq_length, num_layers, mlp_dim, dropout, attention_dropout, out_dim,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),):
        super(Cond_VIT, self).__init__()
        
        
        self.vit_model = vit_model
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        self.hidden_dim = hidden_dim

        # 定义嵌入层
        self.condition_embedding = nn.Embedding(num_attrs, hidden_dim)

        self.cond_encoder = Cond_Encoder(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
        
        self.heads = nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=True)
        
    def forward(self, x, attr):
        
        n, c, h, w = x.shape
        p = self.patch_size
        # torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        # torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # print(n, c, h, w, p, n_h, n_w)
        x = self.vit_model.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        # print(self.vit_model.class_token.shape, x.shape)
        batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        for i, block in enumerate(self.vit_model.encoder.layers):
            x = block(x)
        

        condition_embed = self.condition_embedding(attr)
        condition_embed = condition_embed.unsqueeze(1).repeat(1, x.size(1), 1)  # [批次大小, 序列长度, 条件维度]

        x = self.cond_encoder(x, condition_embed)
        
        x = x[:, 0]

        x = self.heads(x)
        
        return x
    
    
# Cond_SWIN(swin_model, 16, 1024, 32, 3, 49, number_layer, 4096, 0.0, 0.0, out_dim)

class Cond_SWIN(nn.Module):
    def __init__(self, swin_model, patch_size, hidden_dim, num_heads, num_attrs,
                 seq_length, num_layers, mlp_dim, dropout, attention_dropout, out_dim,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),):
        super(Cond_SWIN, self).__init__()
        
        
        self.swin_model = swin_model
        # self.patch_size = patch_size
        self.num_heads = num_heads
        
        self.hidden_dim = hidden_dim

        # 定义嵌入层
        self.condition_embedding = nn.Embedding(num_attrs, hidden_dim)

        self.cond_encoder = Cond_Encoder(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
        
        self.permute = Permute([0, 3, 1, 2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        
        self.heads = nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=True)
        
    def forward(self, x, attr):
        
        
        x = self.swin_model.features(x)
        x = self.swin_model.norm(x)
        
        n, h, w, c = x.shape

        x = x.reshape(n, h*w, c)
        
        condition_embed = self.condition_embedding(attr)
        condition_embed = condition_embed.unsqueeze(1).repeat(1, x.size(1), 1)  # [批次大小, 序列长度, 条件维度]
        # print(condition_embed.shape)

        x = self.cond_encoder(x, condition_embed)
        
        x = x.reshape(n, h, w, c)
        
        x = self.permute(x)

        x = self.avgpool(x)

        x = self.flatten(x)
      
        x = self.heads(x)
        
        return x
        
