
import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def add_gaussian_noise(image):
    """
    向图像添加高斯噪声
    :param image: 灰度图像数组
    :return: 添加噪声后的图像
    """
    row, col = image.shape
    mean = 0
    var = 1.5
    sigma = 50
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # 裁剪到 0-255 范围
    return noisy_image.astype(np.uint8)  # 转换为 uint8 类型

def resize_image(image, target_size=(256, 256)):
    """
    调整图像尺寸
    :param image: 输入图像
    :param target_size: 目标尺寸
    :return: 调整尺寸后的图像
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def process_images(origin_dir, clean_dir, noisy_dir):
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
    if not os.path.exists(noisy_dir):
        os.makedirs(noisy_dir)

     # 使用 tqdm 包装循环
    for img_name in tqdm(os.listdir(origin_dir), desc="Processing Images"):
        img_path = os.path.join(origin_dir, img_name)
        if os.path.isfile(img_path):
            # 读取原始灰度图像
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 添加高斯噪声
            noisy_image = add_gaussian_noise(image)

            image = resize_image(image)
            noisy_image = resize_image(noisy_image)

            # 保存原始图像和带噪声的图像
            cv2.imwrite(os.path.join(clean_dir, img_name), image)
            cv2.imwrite(os.path.join(noisy_dir, img_name), noisy_image)

if __name__ == "__main__":
    origin = 'origin'  # 原始图像文件夹路径
    clean_images = 'clean_images'  # 无噪声图像文件夹路径
    noisy_images = 'noisy_images'  # 带噪声图像文件夹路径

    process_images(origin, clean_images, noisy_images)
