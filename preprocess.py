import torch
from torchvision import transforms
from PIL import Image, ImageOps
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert('L')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).to(device)


