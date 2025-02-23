from PIL import Image

# 加载图像
image_path = "/root/autodl-tmp/data/segmentation/Kvasir-SEG-Split/test/masks/cjyzkmjy8evns070165gf9dmq.jpg"  # 替换为你的图像路径
image = Image.open(image_path)

# 获取图像模式（比如 "RGB", "L", "RGBA" 等）
image_mode = image.mode

# 获取图像大小（宽度和高度）
image_size = image.size

# 获取图像的通道数
channel_count = len(image_mode)

# 获取图像的位深度（单通道位数）
bit_depth = image.getbands()  # 返回通道名称

print(f"图像模式: {image_mode}")
print(f"图像大小: {image_size}")
print(f"通道数: {channel_count}")
print(f"位深度: {bit_depth}")

# 如果需要更多属性，可以结合 numpy 来深入分析
import numpy as np

image_array = np.array(image)
print(f"图像数据类型: {image_array.dtype}")
