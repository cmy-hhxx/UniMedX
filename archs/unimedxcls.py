import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from archs.improvedfastkanconv import ImprovedFastKANConvLayer


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 使用KANConv替换原始卷积
        self.fc1 = ImprovedFastKANConvLayer(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = ImprovedFastKANConvLayer(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # 使用KANConv替换原始卷积
        self.conv = ImprovedFastKANConvLayer(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class KANFeaturePyramid(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            ImprovedFastKANConvLayer(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            ImprovedFastKANConvLayer(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [conv(f) for f, conv in zip(features, self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')

        outputs = [conv(lat) for lat, conv in zip(laterals, self.fpn_convs)]
        return outputs


class ImprovedClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 添加额外的特征处理层
        self.feature_processing = nn.Sequential(
            ImprovedFastKANConvLayer(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            ImprovedFastKANConvLayer(in_channels // 2, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(in_channels // 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.feature_processing(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ImprovedClassifier(nn.Module):
    def __init__(self, num_classes=19, pretrained=True):
        super().__init__()

        # 保持ResNet50作为backbone
        backbone = resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 256
        self.layer2 = backbone.layer2  # 512
        self.layer3 = backbone.layer3  # 1024
        self.layer4 = backbone.layer4  # 2048

        # 使用KAN版本的FPN
        self.fpn = KANFeaturePyramid([256, 512, 1024, 2048], 256)

        # CBAM注意力模块(已包含KANConv)
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(256)

        # 特征融合层
        self.fusion_conv = nn.Sequential(
            ImprovedFastKANConvLayer(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # 添加Dropout
            ImprovedFastKANConvLayer(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)  # 添加Dropout
        )

        # 改进的分类头
        self.classifier = ImprovedClassificationHead(256, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # Backbone特征提取
        x = self.layer0(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # FPN特征金字塔
        fpn_features = self.fpn([c1, c2, c3, c4])

        # 应用CBAM注意力
        p1 = self.cbam1(fpn_features[0])
        p2 = self.cbam2(fpn_features[1])
        p3 = self.cbam3(fpn_features[2])
        p4 = self.cbam4(fpn_features[3])

        # 特征融合
        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=True)
        p3_up = F.interpolate(p3, size=p1.shape[-2:], mode='bilinear', align_corners=True)
        p4_up = F.interpolate(p4, size=p1.shape[-2:], mode='bilinear', align_corners=True)

        fused_features = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
        fused_features = self.fusion_conv(fused_features)

        # 分类预测
        logits = self.classifier(fused_features)

        if not self.training:
            logits = torch.sigmoid(logits)

        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ImprovedFastKANConvLayer):
                # 初始化 KANConv 内部的 spline_conv
                if hasattr(m, 'spline_conv'):
                    if hasattr(m.spline_conv, 'weight'):
                        nn.init.kaiming_normal_(m.spline_conv.weight, mode='fan_out', nonlinearity='relu')
                    if hasattr(m.spline_conv, 'bias') and m.spline_conv.bias is not None:
                        nn.init.constant_(m.spline_conv.bias, 0)
                # 初始化 base_conv（如果存在）
                if hasattr(m, 'base_conv'):
                    if hasattr(m.base_conv, 'weight'):
                        nn.init.kaiming_normal_(m.base_conv.weight, mode='fan_out', nonlinearity='relu')
                    if hasattr(m.base_conv, 'bias') and m.base_conv.bias is not None:
                        nn.init.constant_(m.base_conv.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
