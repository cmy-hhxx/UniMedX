
import os
import shutil
import logging
from pathlib import Path
import random


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_val_directories(base_path):
    """创建验证集目录"""
    val_dirs = [
        os.path.join(base_path, "val/images"),
        os.path.join(base_path, "val/masks")
    ]

    for directory in val_dirs:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"创建验证集目录: {directory}")


def move_to_validation_set(base_path, validation_ratio=0.1, random_seed=42):
    """
    从训练集中移动指定比例的数据到验证集

    Args:
        base_path: 数据集根目录
        validation_ratio: 验证集比例，默认0.1（10%）
        random_seed: 随机种子，确保可重复性
    """
    # 设置随机种子
    random.seed(random_seed)

    # 创建验证集目录
    create_val_directories(base_path)

    # 训练集图像目录
    train_images_dir = os.path.join(base_path, "train/images")
    train_masks_dir = os.path.join(base_path, "train/masks")

    # 验证集目录
    val_images_dir = os.path.join(base_path, "val/images")
    val_masks_dir = os.path.join(base_path, "val/masks")

    # 获取训练集图像列表
    train_images = [f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))]

    # 计算要移动的文件数量
    num_val_samples = int(len(train_images) * validation_ratio)
    logging.info(f"训练集总样本数: {len(train_images)}")
    logging.info(f"将移动 {num_val_samples} 个样本到验证集")

    # 随机选择文件
    val_samples = random.sample(train_images, num_val_samples)

    # 移动文件到验证集
    moved_count = 0
    for filename in val_samples:
        try:
            # 移动图像
            image_src = os.path.join(train_images_dir, filename)
            image_dst = os.path.join(val_images_dir, filename)

            # 移动对应的mask
            mask_src = os.path.join(train_masks_dir, filename)
            mask_dst = os.path.join(val_masks_dir, filename)

            # 确保对应的mask文件存在
            if os.path.exists(mask_src):
                # 使用shutil.move进行剪切操作
                shutil.move(image_src, image_dst)
                shutil.move(mask_src, mask_dst)
                moved_count += 1
                logging.info(f"移动文件到验证集: {filename}")
            else:
                logging.warning(f"未找到对应的mask文件: {filename}")

        except Exception as e:
            logging.error(f"移动文件 {filename} 时发生错误: {str(e)}")

    logging.info(f"完成移动。成功移动 {moved_count} 个样本到验证集")
    logging.info(f"验证集比例: {(moved_count / len(train_images)) * 100:.2f}%")


def create_directories(base_path):
    """创建必要的目录结构"""
    directories = [
        os.path.join(base_path, "train/images"),
        os.path.join(base_path, "train/masks"),
        os.path.join(base_path, "test/images"),
        os.path.join(base_path, "test/masks")
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"创建目录: {directory}")


def process_pneumothorax_dataset(source_base_path, target_base_path):
    """处理数据集，提取感染病例"""
    # 源目录路径
    source_images = os.path.join(source_base_path, "png_images")
    source_masks = os.path.join(source_base_path, "png_masks")

    # 确保源目录存在
    if not (os.path.exists(source_images) and os.path.exists(source_masks)):
        raise FileNotFoundError("源目录不存在")

    # 创建目标目录
    create_directories(target_base_path)

    # 处理文件
    processed_count = {"train": 0, "test": 0}

    for filename in os.listdir(source_images):
        # 解析文件名
        try:
            index, dataset_type, label = filename.split('_')[:3]
            label = label.split('.')[0]  # 移除文件扩展名

            # 只处理感染病例 (label = 1)
            if label == '1':
                # 确定目标路径
                if 'train' in dataset_type:
                    target_subdir = 'train'
                elif 'test' in dataset_type:
                    target_subdir = 'test'
                else:
                    logging.warning(f"无法识别数据集类型: {filename}")
                    continue

                # 复制图片和mask
                image_source = os.path.join(source_images, filename)
                mask_source = os.path.join(source_masks, filename)

                image_target = os.path.join(target_base_path, target_subdir, "images", filename)
                mask_target = os.path.join(target_base_path, target_subdir, "masks", filename)

                if os.path.exists(mask_source):
                    shutil.copy2(image_source, image_target)
                    shutil.copy2(mask_source, mask_target)
                    processed_count[target_subdir] += 1
                    logging.info(f"已复制: {filename} 到 {target_subdir}")
                else:
                    logging.warning(f"未找到对应的mask文件: {filename}")

        except Exception as e:
            logging.error(f"处理文件 {filename} 时发生错误: {str(e)}")

    print(f"处理完成。训练集: {processed_count['train']} 个文件, 测试集: {processed_count['test']} 个文件")


def main():
    """主函数"""
    setup_logging()

    # 设置数据集根目录
    base_path = "/root/autodl-tmp/data/segmentation/Mos"  # 请修改为实际的数据集根目录路径

    try:
        move_to_validation_set(base_path, validation_ratio=0.1)
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")

    # # 设置源目录和目标目录路径
    # source_base_path = "/root/autodl-tmp/pneumothorax"  # 请修改为实际的源目录路径
    # target_base_path = "/root/autodl-tmp/pneumothorax-split"  # 请修改为实际的目标目录路径
    #
    # try:
    #     process_pneumothorax_dataset(source_base_path, target_base_path)
    # except Exception as e:
    #     logging.error(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()
