import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

class CategoricalEmbedding(nn.Module):
    def __init__(self, cat_dims, embed_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=embed_dim)
            for dim in cat_dims
        ])

    def forward(self, x_cat):
        embeds = []
        for i in range(x_cat.shape[1]):
            embed = self.embeddings[i](x_cat[:, i])
            embeds.append(embed)
        return torch.cat(embeds, dim=1)


class ContinuousNormalization(nn.Module):
    def __init__(self, num_cont_features, embed_dim):
        super().__init__()
        self.fc = nn.Linear(num_cont_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x_cont):
        x = self.fc(x_cont)
        x = self.bn(x)
        return x


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, embed_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :].unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


class TabTransformer(nn.Module):
    def __init__(self,
                 cat_cols,  # 类别特征列名列表，用于区分、统计维度
                 num_cont_features,  # 连续特征数量
                 cat_embed_dim=64,
                 num_heads=8,
                 num_layers=4,
                 ff_hidden_dim=256,
                 max_seq_len=100,
                 num_classes=2,
                 extract_features=True):
        super().__init__()
        self.cat_cols = cat_cols
        self.cat_dims = [2, 7, 6, 2, 2, 2, 2]  #每种类别特征含有的类别数
        self.cat_embed_dim = cat_embed_dim
        self.cat_embed = CategoricalEmbedding(self.cat_dims, cat_embed_dim)
        self.cont_norm = ContinuousNormalization(num_cont_features, cat_embed_dim)
        # 确保注意力维度匹配
        assert cat_embed_dim % num_heads == 0, "cat_embed_dim must be divisible by num_heads"

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(
                MultiheadAttention(cat_embed_dim, num_heads, batch_first=True)
            )
        self.num_classes = num_classes
        # self.spatial_pe = SpatialPositionalEncoding(max_seq_len, cat_embed_dim)
        self.extract_features = extract_features

        self.attention_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(
                MultiheadAttention(cat_embed_dim, num_heads, batch_first=True)
            )
            self.ff_layers.append(
                FeedForward(cat_embed_dim, ff_hidden_dim)
            )

        if extract_features:
            self.feature_projection = None  # 动态初始化
        else:
            self.classifier = None

    def forward(self, x_cat, x_cont):
        # 检查类别特征索引有效性
        for i in range(x_cat.shape[1]):
            if (x_cat[:, i] >= self.cat_dims[i]).any():
                # print(f"Invalid cat index in batch: max={x_cat[:, i].max()}, dim={self.cat_dims[i]}")
                # 可选：修正越界索引
                x_cat[:, i] = torch.clamp(x_cat[:, i], max=self.cat_dims[i] - 1)
        x_cat_embed = self.cat_embed(x_cat)
        x_cont_embed = self.cont_norm(x_cont)

        x_combined = torch.cat([x_cat_embed, x_cont_embed], dim=1)
        seq_len = x_combined.shape[1] // self.cat_embed.embeddings[0].embedding_dim
        x_combined = x_combined.reshape(-1, seq_len, self.cat_embed.embeddings[0].embedding_dim)

        if not hasattr(self, 'spatial_pe') or self.spatial_pe.pe.shape[0] != seq_len:
            self.spatial_pe = SpatialPositionalEncoding(seq_len, self.cat_embed.embeddings[0].embedding_dim).to(
                x_combined.device)

        x = self.spatial_pe(x_combined)

        for attn, ff in zip(self.attention_layers, self.ff_layers):
            attn_out, _ = attn(x, x, x)
            x = x + attn_out
            x = F.layer_norm(x, x.shape[1:])

            ff_out = ff(x)
            x = x + ff_out
            x = F.layer_norm(x, x.shape[1:])

        x_flat = x.reshape(x.shape[0], -1)

        if self.extract_features:
            # 动态初始化特征投影层
            if self.feature_projection is None:
                input_dim = x_flat.shape[1]  # 例如512
                self.feature_projection = nn.Linear(input_dim, 1024).to(x_flat.device)
            features = self.feature_projection(x_flat)
            return features
        else:
            # 分类模式
            if self.classifier is None:
                input_dim = x_flat.shape[1]
                self.classifier = nn.Sequential(
                    nn.Linear(input_dim, self.cat_embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.cat_embed_dim * 2, self.num_classes)
                ).to(x_flat.device)
            logits = self.classifier(x_flat)
            return logits