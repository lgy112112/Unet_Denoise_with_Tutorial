
import os
from torch.utils.data import Dataset, DataLoader
from preprocess import load_image

class ImageDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        # 假设 noisy_dir 和 clean_dir 中的文件是一一对应的
        self.file_names = [f for f in os.listdir(noisy_dir) if os.path.isfile(os.path.join(noisy_dir, f))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        noisy_img_path = os.path.join(self.noisy_dir, self.file_names[idx])
        clean_img_path = os.path.join(self.clean_dir, self.file_names[idx])

        noisy_image = load_image(noisy_img_path)
        clean_image = load_image(clean_img_path)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image

# 使用数据集
dataset = ImageDataset(noisy_dir='noisy_images', clean_dir='clean_images')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
