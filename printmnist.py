# 导入 PyTorch 库
import torch
import torchvision
from torchvision import transforms

# 下载 MNIST 数据集
# mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(
        root='./datacifar', train=True, download=True, transform=transform
    )
# 创建数据加载器
data_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# 获取四张图片及其对应的标签
data_iter = iter(data_loader)
images, labels = data_iter.next()

# 显示四张图片
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 6, figsize=(12, 3))
for i in range(6):
    image = images[i].numpy().squeeze()
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {labels[i]}')
    axes[i].axis('off')

plt.show()