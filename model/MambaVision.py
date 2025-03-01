import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth drop path (from timm)"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MambaVisionMixer(nn.Module):
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
        # Implement actual Mamba logic here
        return x  # Simplified for demonstration


class ConvBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels, num_blocks, block_type='conv', drop_path_rates=[]):
        super().__init__()
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            if block_type == 'conv':
                self.blocks.append(ConvBlock(in_channels, drop_path_rates[i]))
            elif block_type == 'attention':
                # Add attention blocks implementation
                pass

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


class MambaVision(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 80, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU()
        )

        self.levels = nn.ModuleList([
            MambaVisionLayer(80, 160, num_blocks=1, block_type='conv'),
            MambaVisionLayer(160, 320, num_blocks=3, block_type='conv'),
            MambaVisionLayer(320, 640, num_blocks=8, block_type='hybrid'),
            MambaVisionLayer(640, 640, num_blocks=4, block_type='attention')
        ])

        # Head
        self.norm = nn.BatchNorm2d(640)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(640, 5),
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