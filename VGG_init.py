
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:23].eval()
        # 修改第一个卷积层以接受单通道输入
        first_conv_layer = vgg[0]
        new_first_conv_layer = nn.Conv2d(1, 64, 3, 1, 1)
        new_first_conv_layer.weight.data = torch.mean(first_conv_layer.weight.data, dim=1, keepdim=True)
        vgg[0] = new_first_conv_layer
        self.vgg = vgg
        self.loss = nn.MSELoss()  # 初始化损失函数

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.loss(x_vgg, y_vgg)
