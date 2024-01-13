import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from unet import UNet

# 设定随机种子
torch.manual_seed(1)
# 设定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = UNet()  # 确保 UNet 类已经定义
model.load_state_dict(torch.load('./unet_model.pth', map_location=device))
model.to(device)
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 从 /content/clean_images 随机选择一张图片
clean_images_path = './clean_images'
selected_image_name = random.choice(os.listdir(clean_images_path))
clean_image_path = os.path.join(clean_images_path, selected_image_name)

# 从 /content/noisy_images 获取对应的带噪声图片
noisy_images_path = './noisy_images'
noisy_image_path = os.path.join(noisy_images_path, selected_image_name)

# 加载图像
clean_image = Image.open(clean_image_path).convert('L')
noisy_image = Image.open(noisy_image_path).convert('L')

# 预处理图像并添加 batch 维度
input_image = transform(noisy_image).unsqueeze(0).to(device)

# 使用模型进行预测
with torch.no_grad():
    predicted_image = model(input_image)
    predicted_image = predicted_image.squeeze(0).cpu()

# 反标准化
predicted_image = predicted_image * 0.5 + 0.5

# 可视化
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Clean Image")
plt.imshow(clean_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Predicted Image")
plt.imshow(predicted_image.numpy()[0], cmap='gray')
plt.axis('off')

plt.show()
