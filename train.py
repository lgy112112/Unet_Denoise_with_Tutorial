
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet
from data_setting import ImageDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

# 设备配置（使用 GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例
model = UNet().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=0.0001)
# 创建感知损失函数实例
perceptual_loss = PerceptualLoss().to(device)
# 准备数据集
train_dataset = ImageDataset(noisy_dir='noisy_images', clean_dir='clean_images')
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 训练模型
num_epochs = 30  # 您可以调整这个值
for epoch in range(num_epochs):
    # 使用 tqdm 包装训练数据加载器
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
        for i, (noisy_imgs, clean_imgs) in tepoch:

            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # 前向传播
            outputs = model(noisy_imgs)
            loss = perceptual_loss(outputs, clean_imgs)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新 tqdm 进度条
            tepoch.set_postfix(loss=loss.item())

torch.save(model.state_dict(), 'unet_model.pth')
