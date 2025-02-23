import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, img_size=(352, 352)):
        """
        Args:
            root_dir (str): 数据集根目录
            phase (str): 'train' 或 'test'
            transform: albumentations转换
            img_size (tuple): 目标图像大小
        """
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.img_size = img_size

        # 设置图像和掩码目录
        self.image_dir = os.path.join(root_dir, phase, 'images')
        self.mask_dir = os.path.join(root_dir, phase, 'masks')

        # 获取图像和掩码文件列表
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(('.jpg', '.png'))])

        # 验证图像和掩码数量匹配
        assert len(self.images) == len(self.masks), \
            f"Number of images ({len(self.images)}) and masks ({len(self.masks)}) don't match"

        print(f"Found {len(self.images)} images in {phase} set")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 构建文件路径
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # 加载图像和掩码
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图

        # 转换为numpy数组
        image = np.array(image)
        mask = np.array(mask)

        # 确保mask为二值图像
        mask = (mask > 128).astype(np.float32)

        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            # 确保mask维度正确 (H, W) -> (1, H, W)
            if isinstance(mask, torch.Tensor) and len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        return {
            'image': image,
            'mask': mask,
            'image_path': img_path
        }


def get_transforms(phase, img_size=(352, 352)):
    """
    获取数据转换pipeline
    Args:
        phase (str): 'train' 或 'test'
        img_size (tuple): 目标图像大小
    """
    # if phase == 'train':
    #     return A.Compose([
    #         A.Resize(height=img_size[0], width=img_size[1]),
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.5),
    #         A.RandomRotate90(p=0.5),
    #         A.ShiftScaleRotate(
    #             shift_limit=0.2,
    #             scale_limit=0.2,
    #             rotate_limit=30,
    #             p=0.5
    #         ),
    #         A.OneOf([
    #             A.ElasticTransform(
    #                 alpha=120,
    #                 sigma=120 * 0.05,
    #                 alpha_affine=120 * 0.03,
    #                 p=0.5
    #             ),
    #             A.GridDistortion(p=0.5),
    #             A.OpticalDistortion(
    #                 distort_limit=1,
    #                 shift_limit=0.5,
    #                 p=0.5
    #             ),
    #         ], p=0.3),
    #         A.OneOf([
    #             A.GaussNoise(p=0.5),
    #             A.RandomBrightnessContrast(p=0.5),
    #             A.ColorJitter(p=0.5),
    #         ], p=0.3),
    #         A.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225],
    #             max_pixel_value=255.0,
    #             p=1.0
    #         ),
    #         ToTensorV2(),
    #     ], p=1.0)
    if phase == 'train':
        return A.Compose([
            # 基础大小调整
            A.Resize(height=img_size[0], width=img_size[1]),

            # 随机裁剪，处理小目标
            A.RandomResizedCrop(
                height=img_size[0],
                width=img_size[1],
                scale=(0.8, 1.0),
                ratio=(0.75, 1.333),
                p=0.5
            ),

            # 基础几何变换，提高概率
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            A.RandomRotate90(p=0.7),

            # 位移、缩放、旋转
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=30,
                border_mode=4,  # BORDER_REFLECT_101
                p=0.7
            ),

            # 弹性变换，降低强度
            A.OneOf([
                A.ElasticTransform(
                    alpha=60,  # 降低强度
                    sigma=60 * 0.05,
                    p=0.3
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(
                    distort_limit=0.3,
                    shift_limit=0.3,
                    p=0.5
                ),
            ], p=0.3),

            # 针对小目标区域的特定增强
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8,
                    max_height=20,
                    max_width=20,
                    min_holes=5,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.5
                ),
                A.GridDropout(
                    ratio=0.3,
                    unit_size_min=10,
                    unit_size_max=40,
                    holes_number_x=4,
                    holes_number_y=4,
                    p=0.5
                ),
            ], p=0.3),

            # 噪声和颜色增强
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
            ], p=0.3),

            # 模糊和锐化
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            ], p=0.3),

            # 标准化
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(),
        ], p=1.0)

    else:
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(),
        ], p=1.0)


# 测试时增强（TTA）的实现
def get_tta_transforms(img_size=(352, 352)):
    """
    获取测试时增强的transforms
    """
    tta_transforms = []
    # 原始图像
    tta_transforms.append(
        A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
    )

    # 水平翻转
    tta_transforms.append(
        A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
    )

    # 垂直翻转
    tta_transforms.append(
        A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.VerticalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
    )

    return tta_transforms



def get_test_dataloaders(
        data_dir,
        batch_size=1,
        num_workers=0,
        img_size=(256, 256)
):
    # 创建训练集
    test_dataset = SegDataset(
        root_dir=data_dir,
        phase='test',
        transform=get_transforms('test', img_size),
        img_size=img_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return test_loader


def get_dataloaders(
        data_dir,
        batch_size=1,
        num_workers=0,
        img_size=(256, 256)
):
    """
    创建训练和测试数据加载器
    Args:
        data_dir (str): 数据集根目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        img_size (tuple): 目标图像大小
    """
    # 创建训练集
    train_dataset = SegDataset(
        root_dir=data_dir,
        phase='train',
        transform=get_transforms('train', img_size),
        img_size=img_size
    )

    # 创建测试集
    val_dataset = SegDataset(
        root_dir=data_dir,
        phase='val',
        transform=get_transforms('val', img_size),
        img_size=img_size
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader




import matplotlib.pyplot as plt
import torchvision.transforms as T


def visualize_batch(batch, num_samples=4):
    images = batch['image'][:num_samples]
    masks = batch['mask'][:num_samples]

    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean

    # 将图像从 [0, 1] 转换到 [0, 255]
    images = (images * 255).byte()

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))

    for i in range(num_samples):
        # 显示图像
        axes[i, 0].imshow(T.ToPILImage()(images[i]))
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis('off')

        # 显示掩码
        axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Mask {i + 1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 测试代码
    data_dir = "/root/autodl-tmp/data/segmentation/Kvasir-SEG-Split"  # 替换为实际的数据集路径
    batch_size = 4
    img_size = (256, 256)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size
    )
    test_loader =get_test_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of val batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # 检查一个批次的数据
    for batch in train_loader:
        images = batch['image']
        masks = batch['mask']
        print(f"Batch image shape: {images.shape}")
        print(f"Batch mask shape: {masks.shape}")
        print(f"Image data range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask data range: [{masks.min():.3f}, {masks.max():.3f}]")
        visualize_batch(batch)
        break