import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ColonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): 包含图像标签的CSV文件路径
            img_dir (string): 所有图像文件所在的目录
            transform (callable, optional): 可选的图像转换操作
        """
        # 读取CSV文件，移除包含NaN值的行
        self.annotations = pd.read_csv(csv_file)

        # 移除包含NaN值的行
        self.annotations = self.annotations.dropna(subset=['img_id'])

        # 确保img_id列为字符串类型
        self.annotations['img_id'] = self.annotations['img_id'].astype(str)

        # 重置索引
        self.annotations = self.annotations.reset_index(drop=True)

        self.img_dir = img_dir

        # 确保肿瘤标签为数值类型
        self.annotations['tumor'] = self.annotations['tumor'].astype(np.float32)

        # 验证所有图像文件是否存在
        self.validate_images()

        # 如果没有指定transform，则使用默认的转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像大小
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        # 添加疾病标签映射
        self.disease_labels = list(self.annotations.columns)[2:]

    def validate_images(self):
        """验证所有图像文件是否存在，移除不存在的记录"""
        valid_indices = []
        for idx, row in self.annotations.iterrows():
            img_path = os.path.join(self.img_dir, row['img_id'])
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                print(f"Warning: Image not found: {img_path}")

        self.annotations = self.annotations.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.annotations)

    def get_label_names(self):
        """返回疾病标签列表"""
        return self.disease_labels


    def __getitem__(self, idx):
        """
        获取指定索引的样本
        Args:
            idx (int): 样本索引
        Returns:
            tuple: (image, label, img_id) 包含图像、标签和图像ID
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径和ID
        img_id = str(self.annotations.iloc[idx]['img_id'])
        img_path = os.path.join(self.img_dir, img_id)

        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise

        # 应用图像转换
        if self.transform:
            image = self.transform(image)

        # 获取标签
        label = self.annotations.iloc[idx]['tumor']
        label = torch.tensor(label, dtype=torch.float32)

        return image, label, img_id


def get_test_colon_dataloader(csv_path, img_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    创建测试数据加载器
    Args:
        csv_path (string): CSV文件路径
        img_dir (string): 图像目录路径
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 数据加载的进程数
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = ColonDataset(csv_path, img_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


def get_colon_dataloader(csv_path, img_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    创建训练数据加载器
    Args:
        csv_path (string): CSV文件路径
        img_dir (string): 图像目录路径
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 数据加载的进程数
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = ColonDataset(csv_path, img_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


# 使用示例
if __name__ == '__main__':
    # 设置路径
    csv_path = '/root/autodl-tmp/data/classification/MedFMC_train/colon/colon_train.csv'
    img_dir = '/root/autodl-tmp/data/classification/MedFMC_train/colon/images'

    # 创建数据集实例
    dataset = ColonDataset(csv_path, img_dir)
    print(f"数据集大小: {len(dataset)}")
    print(f"疾病标签: {dataset.get_label_names()}")


    # 获取一个样本
    image, label, img_id = dataset[0]
    print(f"图像尺寸: {image.shape}")
    print(f"标签: {label}")
    print(f"图像ID: {img_id}")

    # 创建数据加载器
    dataloader = get_test_colon_dataloader(csv_path, img_dir)

    # 测试数据加载器
    for batch_images, batch_labels, batch_img_ids in dataloader:
        print(f"批次图像尺寸: {batch_images.shape}")
        print(f"批次标签尺寸: {batch_labels.shape}")
        print(f"批次图像ID: {batch_img_ids}")
        break
