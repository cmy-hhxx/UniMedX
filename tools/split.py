import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np


def split_dataset(source_dir, train_ratio=0.8, random_state=42):
    # 创建训练集和测试集的主目录
    base_dir = os.path.dirname(source_dir)
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # 确保输出目录存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历源目录中的每个类别
    for class_name in ['COVID', 'non-COVID']:
        # 源类别目录
        class_dir = os.path.join(source_dir, class_name)

        # 创建对应的训练集和测试集子目录
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # 获取该类别下所有图像文件
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.dcm'))]

        # 划分数据集
        train_files, test_files = train_test_split(
            image_files,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True
        )

        # 复制文件到相应目录
        for file_name in train_files:
            src = os.path.join(class_dir, file_name)
            dst = os.path.join(train_class_dir, file_name)
            shutil.copy2(src, dst)

        for file_name in test_files:
            src = os.path.join(class_dir, file_name)
            dst = os.path.join(test_class_dir, file_name)
            shutil.copy2(src, dst)

        # 打印每个类别的划分统计信息
        print(f"\n类别 {class_name} 的数据划分：")
        print(f"总数：{len(image_files)}")
        print(f"训练集：{len(train_files)} ({len(train_files) / len(image_files) * 100:.1f}%)")
        print(f"测试集：{len(test_files)} ({len(test_files) / len(image_files) * 100:.1f}%)")


def main():
    # 源数据集路径
    source_dir = "/root/autodl-tmp/data/classification/ct_covid"

    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误：源目录 {source_dir} 不存在！")
        return

    print("开始划分数据集...")
    split_dataset(source_dir)
    print("\n数据集划分完成！")

    # 打印最终的目录结构
    print("\n生成的目录结构：")
    for root, dirs, files in os.walk("./"):
        if ".git" in dirs:
            dirs.remove(".git")  # 忽略git目录
        level = root.replace("./", "").count(os.sep)
        indent = "│   " * level
        print(f"{indent}├── {os.path.basename(root)}/")


if __name__ == "__main__":
    main()
