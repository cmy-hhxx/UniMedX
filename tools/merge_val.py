import os
import shutil


def merge_valid_to_train(dataset_path):
    # 获取验证集和训练集的路径
    if 'chest_ctscan' in dataset_path:
        valid_dir = os.path.join(dataset_path, 'valid')
        train_dir = os.path.join(dataset_path, 'train')
    else:  # chest_xray
        valid_dir = os.path.join(dataset_path, 'val')
        train_dir = os.path.join(dataset_path, 'train')

    # 确保验证集目录存在
    if not os.path.exists(valid_dir):
        print(f"验证集目录不存在: {valid_dir}")
        return

    # 遍历验证集中的所有类别目录
    for class_name in os.listdir(valid_dir):
        valid_class_path = os.path.join(valid_dir, class_name)
        train_class_path = os.path.join(train_dir, class_name)

        # 确保训练集中存在对应的类别目录
        if not os.path.exists(train_class_path):
            os.makedirs(train_class_path)

        # 遍历验证集类别目录中的所有图像
        for img_name in os.listdir(valid_class_path):
            src_path = os.path.join(valid_class_path, img_name)
            dst_path = os.path.join(train_class_path, img_name)

            # 移动图像到训练集
            try:
                shutil.move(src_path, dst_path)
                print(f"已移动: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"移动文件失败: {str(e)}")

    # 删除空的验证集目录
    try:
        shutil.rmtree(valid_dir)
        print(f"已删除验证集目录: {valid_dir}")
    except Exception as e:
        print(f"删除验证集目录失败: {str(e)}")


def main():
    # 数据集路径
    chest_ctscan_path = "/root/autodl-tmp/data/classification/chest_ctscan"
    chest_xray_path = "/root/autodl-tmp/data/classification/chest_xray"

    # 处理两个数据集
    print("处理 chest_ctscan 数据集...")
    merge_valid_to_train(chest_ctscan_path)

    print("\n处理 chest_xray 数据集...")
    merge_valid_to_train(chest_xray_path)


if __name__ == "__main__":
    main()
