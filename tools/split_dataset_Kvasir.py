import os
import shutil
import random
from sklearn.model_selection import train_test_split


def create_dirs(base_path):

    """创建必要的目录结构"""
    for split in ['train', 'val', 'test']:
        for type in ['images', 'masks']:
            os.makedirs(os.path.join(base_path, split, type), exist_ok=True)


def split_dataset(data_dir, output_dir, train_ratio=0.8, val_ratio=0.05, test_ratio=0.15):
    """
    划分数据集并复制文件到相应目录

    参数:
    data_dir: 原始数据目录
    output_dir: 输出目录
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    test_ratio: 测试集比例
    """
    # 确保比例和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5

    # 创建目录结构
    create_dirs(output_dir)

    # 获取所有图像文件名
    image_files = os.listdir(os.path.join(data_dir, 'images'))

    # 首先划分出训练集
    train_files, temp_files = train_test_split(
        image_files,
        train_size=train_ratio,
        random_state=42
    )

    # 从剩余数据中划分验证集和测试集
    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_ratio / (val_ratio + test_ratio),
        random_state=42
    )

    # 复制文件到相应目录
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, files in splits.items():
        print(f"{split_name} set size: {len(files)}")
        for filename in files:
            # 复制图像
            src_image = os.path.join(data_dir, 'images', filename)
            dst_image = os.path.join(output_dir, split_name, 'images', filename)
            shutil.copy2(src_image, dst_image)

            # 复制对应的掩膜
            src_mask = os.path.join(data_dir, 'masks', filename)
            dst_mask = os.path.join(output_dir, split_name, 'masks', filename)
            shutil.copy2(src_mask, dst_mask)


if __name__ == '__main__':
    # 设置路径
    data_dir = '/root/autodl-tmp/data/segmentation/cor/train'  # 原始数据目录
    output_dir = '/root/autodl-tmp/data/segmentation/cor-split'  # 输出目录

    # 执行划分
    split_dataset(data_dir, output_dir)

    # 打印每个集合的大小
    for split in ['train', 'val', 'test']:
        n_images = len(os.listdir(os.path.join(output_dir, split, 'images')))
        print(f"Number of images in {split} set: {n_images}")
