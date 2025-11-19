import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample:
            out = F.avg_pool2d(out, 2)
            residual = F.avg_pool2d(residual, 2)
            
        return F.leaky_relu(out + residual, 0.2)

class StyleEncoder(nn.Module):
    def __init__(self, in_channels=80, hidden_channels=128, out_channels=256):
        super().__init__()
        self.conv_in = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden_channels, hidden_channels, downsample=True),
            ResBlock(hidden_channels, hidden_channels*2, downsample=True),
            ResBlock(hidden_channels*2, hidden_channels*4, downsample=True),
            ResBlock(hidden_channels*4, hidden_channels*4, downsample=True),
        ])
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_channels*4, out_channels)
        
    def forward(self, x):
        # x: (B, n_mels, T)
        x = x.unsqueeze(1) # (B, 1, n_mels, T)
        x = self.conv_in(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.pool(x).squeeze(-1).squeeze(-1) # (B, hidden*4)
        x = self.proj(x)
        return x

