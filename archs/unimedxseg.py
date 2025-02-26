import torch
from torch import nn
import torch.nn.functional as F
from archs.fastkanconv import FastKANConvLayer
from archs.improvedfastkanconv import ImprovedFastKANConvLayer
import math


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv(x)
#         return self.sigmoid(x)

#
# class WindowQKVAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16, window_size=8, heads=4):
#         super().__init__()
#         self.in_channels = in_channels
#         self.reduction = reduction
#         self.window_size = window_size
#         self.heads = heads
#         self.scale = (in_channels // reduction // heads) ** -0.5
#
#         # QKV projections
#         self.to_qkv = nn.Conv2d(in_channels, (in_channels // reduction) * 3, 1, bias=False)
#
#         # Output projection
#         self.to_out = nn.Sequential(
#             nn.Conv2d(in_channels // reduction, in_channels, 1),
#             nn.BatchNorm2d(in_channels)
#         )
#
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def window_partition(self, x):
#         B, C, H, W = x.shape
#         h_windows = H // self.window_size
#         w_windows = W // self.window_size
#
#         # Reshape to windows
#         x = x.view(B, C, h_windows, self.window_size, w_windows, self.window_size)
#         x = x.permute(0, 2, 4, 1, 3, 5)
#         x = x.reshape(-1, C, self.window_size, self.window_size)
#         return x
#
#     def window_reverse(self, windows, H, W):
#         B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
#         h_windows = H // self.window_size
#         w_windows = W // self.window_size
#
#         # Reshape back
#         x = windows.view(B, h_windows, w_windows, -1, self.window_size, self.window_size)
#         x = x.permute(0, 3, 1, 4, 2, 5)
#         x = x.reshape(B, -1, H, W)
#         return x
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         # Pad input if needed
#         pad_h = (self.window_size - H % self.window_size) % self.window_size
#         pad_w = (self.window_size - W % self.window_size) % self.window_size
#         if pad_h > 0 or pad_w > 0:
#             x = F.pad(x, (0, pad_w, 0, pad_h))
#
#         # Window partition
#         x_windows = self.window_partition(x)
#
#         # QKV projections
#         qkv = self.to_qkv(x_windows)  # (B*num_windows, 3*C', Wh, Ww)
#         qkv = qkv.reshape(-1, 3, self.heads, C // self.reduction // self.heads,
#                           self.window_size * self.window_size)
#         q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B*nW, head, dim, Wh*Ww)
#
#         # Compute attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         # Apply attention to value
#         x = (attn @ v)  # (B*nW, head, Wh*Ww, dim)
#         x = x.reshape(-1, C // self.reduction, self.window_size, self.window_size)
#
#         # Output projection
#         x = self.to_out(x)
#
#         # Reverse windows
#         x = self.window_reverse(x, H + pad_h, W + pad_w)
#
#         # Remove padding if added
#         if pad_h > 0 or pad_w > 0:
#             x = x[:, :, :H, :W]
#
#         # Residual connection
#         return x * self.gamma + x


class EnhancedSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels

        # Multi-scale spatial attention
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # Channel attention for spatial features
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 3, 1),  # 3 weights for 3 kernel sizes
            nn.Softmax(dim=1)
        )

        # Learnable parameters
        self.gamma = nn.Parameter(torch.zeros(1))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Generate channel attention weights
        kernel_weights = self.channel_gate(x)

        # Compute spatial statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out, max_out], dim=1)

        # Multi-scale spatial attention
        attn3 = torch.sigmoid(self.conv3(spatial_features))
        attn5 = torch.sigmoid(self.conv5(spatial_features))
        attn7 = torch.sigmoid(self.conv7(spatial_features))

        # Combine multi-scale attention maps with learned weights
        attention_map = (attn3 * kernel_weights[:, 0:1, :, :] +
                         attn5 * kernel_weights[:, 1:2, :, :] +
                         attn7 * kernel_weights[:, 2:3, :, :])

        # Residual attention mechanism
        return x * (1 + self.gamma * attention_map)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ImprovedFastKANConvLayer(in_channels, out_channels // 2, padding=1, kernel_size=3)
        # self.conv1 = nn.Conv2d(in_channels, out_channels // 2, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = ImprovedFastKANConvLayer(out_channels // 2, out_channels, padding=1, kernel_size=3)
        # self.conv2 = nn.Conv2d(out_channels // 2, out_channels, padding=1, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(out_channels)
        # self.sa = WindowQKVAttention(out_channels)
        # self.attention_gate = nn.Parameter(torch.tensor([0.5]))
        #
        # 残差连接
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        se_out = self.se(out)

        sa_out = self.sa(out)
        g = torch.sigmoid(self.attention_gate)
        out = g * se_out + (1 - g) * sa_out

        out += identity
        out = self.relu(out)
        return out


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, device='cuda'):
        super().__init__()
        self.device = device
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, device=self.device)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, device='cuda'):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, device=device)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, device=device)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = ImprovedFastKANConvLayer(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UniMedXSeg(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, device='mps'):
        super(UniMedXSeg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.device = device

        self.channels = [64, 128, 256, 512, 1024]

        self.inc = (DoubleConv(n_channels, 64, device=self.device))

        self.down1 = (Down(self.channels[0], self.channels[1], self.device))
        self.down2 = (Down(self.channels[1], self.channels[2], self.device))
        self.down3 = (Down(self.channels[2], self.channels[3], self.device))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.channels[3], self.channels[4] // factor, self.device))
        self.up1 = (Up(self.channels[4], self.channels[3] // factor, bilinear, self.device))
        self.up2 = (Up(self.channels[3], self.channels[2] // factor, bilinear, self.device))
        self.up3 = (Up(self.channels[2], self.channels[1] // factor, bilinear, self.device))
        self.up4 = (Up(self.channels[1], self.channels[0], bilinear, self.device))
        self.outc = (OutConv(self.channels[0], n_classes))
        self.ds1 = nn.Conv2d(self.channels[3], n_classes, 1)
        self.ds2 = nn.Conv2d(self.channels[2], n_classes, 1)
        self.ds3 = nn.Conv2d(self.channels[1], n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        ds1 = F.interpolate(self.ds1(x), size=x.shape[2:], mode='bilinear')

        x = self.up2(x, x3)
        ds2 = F.interpolate(self.ds2(x), size=x.shape[2:], mode='bilinear')

        x = self.up3(x, x2)
        ds3 = F.interpolate(self.ds3(x), size=x.shape[2:], mode='bilinear')

        x = self.up4(x, x1)
        logits = self.outc(x)

        if self.training:
            return logits, ds1, ds2, ds3
        return logits
