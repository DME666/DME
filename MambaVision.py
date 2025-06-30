import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """随机深度丢弃路径（来自timm）"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """多层感知机模块"""

    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(approximate='none')
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """自注意力模块"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        original_shape = x.shape
        is_4d = len(original_shape) == 4
        if is_4d:
            B, C, H, W = original_shape
            x = x.flatten(2).transpose(1, 2)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if is_4d:
            x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class MambaVisionMixer(nn.Module):
    """MambaVision混合模块（简化演示）"""

    def __init__(self, in_dim, expand_ratio=4):
        super().__init__()
        hidden_dim = int(in_dim * expand_ratio)
        self.in_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.x_proj = nn.Linear(hidden_dim, 36, bias=False)
        self.dt_proj = nn.Linear(20, hidden_dim, bias=True)
        self.out_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.conv1d_x = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, bias=False)
        self.conv1d_z = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, bias=False)

    def forward(self, x):
        # 此处实现实际的Mamba逻辑
        return x  # 简化演示


class ConvBlock(nn.Module):
    """卷积块模块"""

    def __init__(self, in_channels, drop_path_rate=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.drop_path(x)
        x += shortcut
        return x


class MambaVisionLayer(nn.Module):
    """改进后的MambaVision层次模块（下采样统一移至最后）"""

    def __init__(self, in_channels, out_channels, num_blocks, block_type='conv', drop_path_rates=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels
        drop_path_rates = drop_path_rates or [0.0] * num_blocks

        # 构建所有卷积块或注意力块
        for i in range(num_blocks):
            if block_type == 'conv':
                self.blocks.append(ConvBlock(in_channels, drop_path_rates[i]))
            elif block_type == 'attention':
                # 注意力块实现（简化示例）
                attention_block = Attention(in_channels)
                channel_adapter = nn.Conv2d(in_channels, in_channels, kernel_size=1)
                self.blocks.append(nn.Sequential(attention_block, channel_adapter))
            elif block_type == 'hybrid':
                # 混合块逻辑（可根据需求扩展）
                pass

        # 下采样仅在所有块处理完成后执行一次
        # self.channel_adapter = nn.Identity()
        # if num_blocks > 0 and block_type == 'attention':
            # 如果是注意力模块，可能需要调整通道数
        #     self.channel_adapter = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        ) if num_blocks > 0 else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        # 所有块处理完毕后执行下采样
        x = self.downsample(x)
        return x


class MambaVision(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        # 特征提取初始模块（Stem）
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 80, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU()
        )

        # 层次定义（调整下采样位置）
        self.levels = nn.ModuleList([
            # 第一层：1个卷积块 + 下采样
            MambaVisionLayer(80, 160, num_blocks=1, block_type='conv'),
            # 第二层：3个卷积块 + 下采样（下采样移至最后）
            MambaVisionLayer(160, 320, num_blocks=3, block_type='conv'),
            # 第三层：8个混合块 + 下采样
            MambaVisionLayer(320, 640, num_blocks=8, block_type='hybrid'),
            # 第四层：4个注意力块 + 下采样
            MambaVisionLayer(640, 640, num_blocks=4, block_type='attention')
        ])

        # 分类头
        self.norm = nn.BatchNorm2d(640)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(in_features = 640, out_features = 5),
            nn.ReLU(),
            nn.Linear(5, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x).flatten(1)
        x = self.head(x)
        return x