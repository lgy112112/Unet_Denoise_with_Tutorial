import os
import cv2
import numpy as np
import random

def calculate_mse(image1, image2):
    """
    计算两个图像之间的均方误差 (MSE)
    :param image1: 第一个图像
    :param image2: 第二个图像
    :return: 两个图像之间的 MSE
    """
    return np.mean((image1 - image2) ** 2)

def measure_noise_level(folder1, folder2):
    """
    随机选择一个 pgm 图像，比较两个文件夹中相同图像的噪声水平
    :param folder1: 第一个文件夹路径（clean_images）
    :param folder2: 第二个文件夹路径（noisy_images）
    :return: 噪声水平 (MSE)
    """
    pgm_files = [f for f in os.listdir(folder1) if f.endswith('.pgm')]
    if not pgm_files:
        raise ValueError("没有找到 pgm 文件")

    image_name = random.choice(pgm_files)
    print(f"选中的图像: {image_name}")

    path1 = os.path.join(folder1, image_name)
    path2 = os.path.join(folder2, image_name)

    if not os.path.exists(path1) or not os.path.exists(path2):
        raise ValueError("图像文件不存在于提供的文件夹路径中")

    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    if image1.shape != image2.shape:
        raise ValueError("两个图像的尺寸不相同")

    mse = calculate_mse(image1, image2)
    return mse

# 示例使用
folder1 = 'clean_images'
folder2 = 'noisy_images'
noise_level = measure_noise_level(folder1, folder2)
print(f"噪声水平 (MSE): {noise_level}")
