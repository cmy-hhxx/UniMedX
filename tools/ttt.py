import os


def rename_files(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if "mask" in filename:
            # 构建新的文件名
            new_filename = filename.replace("_mask", "")

            # 构建完整的文件路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)

            # 重命名文件
            try:
                os.rename(old_file, new_file)
                print(f'Renamed: {filename} -> {new_filename}')
            except Exception as e:
                print(f'Error renaming {filename}: {str(e)}')


# 使用示例
directory = "/root/autodl-tmp/data/segmentation/Mos/train/masks"  # 替换为你的目录路径
rename_files(directory)
