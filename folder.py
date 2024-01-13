import os

def create_dataset_folders(noisy_dir='noisy_images', clean_dir='clean_images', origin_dir='origin'):
    """
    创建用于存放带噪图像、无噪图像和原始图像的文件夹。
    :param noisy_dir: 存放带噪图像的文件夹路径。
    :param clean_dir: 存放无噪图像的文件夹路径。
    :param origin_dir: 存放原始图像的文件夹路径。
    """
    # 创建原始图像的文件夹
    if not os.path.exists(origin_dir):
        os.makedirs(origin_dir)
        print(f"已创建文件夹: {origin_dir}")
    else:
        print(f"文件夹 {origin_dir} 已存在。")

    # 创建带噪图像的文件夹
    if not os.path.exists(noisy_dir):
        os.makedirs(noisy_dir)
        print(f"已创建文件夹: {noisy_dir}")
    else:
        print(f"文件夹 {noisy_dir} 已存在。")

    # 创建无噪图像的文件夹
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
        print(f"已创建文件夹: {clean_dir}")
    else:
        print(f"文件夹 {clean_dir} 已存在。")

if __name__ == "__main__":
    create_dataset_folders()
