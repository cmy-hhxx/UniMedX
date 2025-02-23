import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Tuple, List, Dict


class BaseDataset(Dataset):
    """基础数据集类"""

    def __init__(self, root_dir: str, split: str, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.samples = []  # 存储(image_path, label, filename)
        self.class_to_idx = {}  # 类别到索引的映射
        self.num_classes = 0  # 类别数量

        # 如果没有指定transform，根据split选择默认的transform
        if transform is None:
            if split == 'train':
                self.transform = self.get_train_transform()
            else:
                self.transform = self.get_valid_transform()
        else:
            self.transform = transform

    def get_train_transform(self):
        """训练集的数据增强"""
        return transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomEqualize(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])

    def get_valid_transform(self):
        """验证集和测试集的基础预处理"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, label, filename = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, filename

    def get_transforms_info(self) -> str:
        """返回当前使用的数据增强信息"""
        transforms_list = []
        for t in self.transform.transforms:
            transforms_list.append(t.__class__.__name__)
        return f"数据集划分: {self.split}\n使用的变换: {', '.join(transforms_list)}"


class ChestCTScanDataset(BaseDataset):
    """胸部CT扫描数据集 - 4个类别"""

    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform)

        # 定义类别
        self.classes = ['adenocarcinoma', 'large.cell.carcinoma', 'normal',
                        'squamous.cell.carcinoma']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = 4  # 明确设置类别数

        # 加载数据
        split_dir = os.path.join(root_dir, 'chest_ctscan', split)
        self._load_samples(split_dir)
    def get_label_names(self):
        """返回标签名称列表"""
        return self.classes

    def _load_samples(self, split_dir: str):
        for class_name in os.listdir(split_dir):
            if class_name.startswith('.'):  # 跳过隐藏文件
                continue

            # 获取基础类别名（去除额外的描述信息）
            base_class = next(c for c in self.classes if c in class_name)
            label = self.class_to_idx[base_class]
            class_dir = os.path.join(split_dir, class_name)

            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, torch.tensor(label), img_name))


class ChestXRayDataset(BaseDataset):
    """胸部X光数据集 - 2个类别"""

    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform)

        # 定义类别
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = 2  # 明确设置类别数

        # 处理验证集命名不一致的问题
        split = 'val' if split == 'valid' else split

        # 加载数据
        split_dir = os.path.join(root_dir, 'chest_xray', split)
        self._load_samples(split_dir)

    def get_label_names(self):
        """返回标签名称列表"""
        return self.classes
    def _load_samples(self, split_dir: str):
        for class_name in os.listdir(split_dir):
            if class_name.startswith('.'):
                continue

            label = self.class_to_idx[class_name]
            class_dir = os.path.join(split_dir, class_name)

            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, torch.tensor(label), img_name))


class CTCovidDataset(BaseDataset):
    """CT新冠检测数据集 - 2个类别"""

    def __init__(self, root_dir: str, split: str = None, transform=None):
        super().__init__(root_dir, split, transform)

        # 定义类别
        self.classes = ['COVID', 'non-COVID']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = 2  # 明确设置类别数

        # 加载数据
        dataset_dir = os.path.join(root_dir, 'ct_covid', split)
        self._load_samples(dataset_dir)
    def get_label_names(self):
        """返回标签名称列表"""
        return self.classes
    def _load_samples(self, dataset_dir: str):
        for class_name in os.listdir(dataset_dir):
            if class_name.startswith('.'):
                continue

            label = self.class_to_idx[class_name]
            class_dir = os.path.join(dataset_dir, class_name)

            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, torch.tensor(label), img_name))


def get_dataloader(dataset_name: str, root_dir: str, split: str,
                   batch_size: int = 32, shuffle: bool = True,
                   num_workers: int = 4) -> torch.utils.data.DataLoader:
    """获取指定数据集的数据加载器"""
    dataset_classes = {
        'chest_ctscan': ChestCTScanDataset,
        'chest_xray': ChestXRayDataset,
        'ct_covid': CTCovidDataset
    }

    dataset_cls = dataset_classes.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"未知的数据集名称: {dataset_name}")

    dataset = dataset_cls(root_dir=root_dir, split=split)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == '__main__':
    # 测试代码
    root_dir = '/root/autodl-tmp/data/classification'

    # 测试各个数据集的类别数
    datasets = {
        'chest_ctscan': ChestCTScanDataset(root_dir, 'train'),
        'chest_xray': ChestXRayDataset(root_dir, 'test'),
        'ct_covid': CTCovidDataset(root_dir, 'train')
    }

    dataloader = get_dataloader('ct_covid', root_dir, 'train')
    for batch_images, batch_labels, batch_filenames in dataloader:
        print(f"批次图像尺寸: {batch_images.shape}")
        print(f"批次标签尺寸: {batch_labels.shape}")
        print(batch_labels)
        print(f"批次文件名: {batch_filenames}")
        break

    # for name, dataset in datasets.items():
    #     print(f"数据集大小: {len(dataset)}")
    #     print(f"{name} 类别数: {dataset.num_classes}")
    #     print(f"{name} 类别映射: {dataset.class_to_idx}")
    #
    #     # 测试一个样本
    #     image, label, filename = dataset[0]
    #     print(f"样本图像尺寸: {image.shape}")
    #     print(f"样本标签: {label}")
    #     print(f"样本文件名: {filename}")
    #     print("-" * 50)
