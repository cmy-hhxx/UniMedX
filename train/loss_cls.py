import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pt = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = -alpha_weight * focal_weight * torch.log(pt + 1e-8)
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target, dim=1)
        union = torch.sum(pred, dim=1) + torch.sum(target, dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth_target = target * self.confidence + 0.5 * self.smoothing
        loss = -(smooth_target * torch.log(pred + 1e-8) +
                 (1 - smooth_target) * torch.log(1 - pred + 1e-8))
        return loss.mean()


class AutoWeightedLoss(nn.Module):
    """自动学习损失权重"""

    def __init__(self, num_losses):
        super().__init__()
        params = torch.ones(num_losses, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class CombinedLoss(nn.Module):
    """组合多个损失函数"""

    def __init__(self, auto_weight=True, focal_weight=1.0,
                 dice_weight=1.0, smooth_weight=0.5):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.smooth_loss = LabelSmoothingLoss()

        self.auto_weight = auto_weight
        if auto_weight:
            self.auto_weight_layer = AutoWeightedLoss(3)
        else:
            self.focal_weight = focal_weight
            self.dice_weight = dice_weight
            self.smooth_weight = smooth_weight

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        smooth = self.smooth_loss(pred, target)

        if self.auto_weight:
            # 自动学习权重
            loss = self.auto_weight_layer([focal, dice, smooth])
        else:
            # 使用固定权重
            loss = (self.focal_weight * focal +
                    self.dice_weight * dice +
                    self.smooth_weight * smooth)

        return loss


# 使用示例
"""
# 在训练代码中替换原有的BCEWithLogitsLoss:
criterion = CombinedLoss(
    num_classes=config['num_classes'],
    auto_weight=True,  # 设置是否使用自动权重
    focal_weight=1.0,  # 手动设置focal loss权重
    dice_weight=1.0,   # 手动设置dice loss权重
    smooth_weight=0.5  # 手动设置label smoothing权重
).cuda()
"""
