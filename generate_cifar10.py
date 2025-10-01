import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import matplotlib.pyplot as plt
seed = 42
random.seed(1)
np.random.seed(1)
num_clients = 20

num_classes = 10
dir_path = "Cifar10/"


# Allocate data to users
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # if check(
    #     config_path,
    #     train_path,
    #     test_path,
    #     num_clients,
    #     num_classes,
    #     niid,
    #     balance,
    #     partition,
    # ):
    #     return

    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=False, download=True, transform=transform
    )

    # 获取四张图片及其对应的标签
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    # 显示四张图片
    # 获取批次中的图片数量
    num_images = images.size(0)

    # 确保我们不会尝试访问超出范围的索引
    num_to_show = min(num_images, 6)
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    for i in range(num_to_show):
        ax = axes[i // 3, i % 3]

        ax.imshow(images[i].permute(1, 2, 0))  # 将张量的维度从(C, H, W)改为(H, W, C)以进行可视化
        ax.set_title(f'Class {labels[i].item()}')
        ax.axis('off')

        # 如果批次中的图片少于6张，则隐藏剩余的axes
    for i in range(num_to_show, 6):
        ax = axes[i // 3, i % 3]
        ax.axis('off')
        ax.set_visible(False)

    plt.tight_layout()

    plt.show()



    classes = trainset.classes
    print(classes)
    n_classes = len(classes)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False
    )

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
    )

    label_distribution = [[] for _ in range(n_classes)]

    for client_id, label_num in enumerate(statistic):

        for label in label_num:
            for i in range(label[1]):
                label_distribution[label[0]].append(client_id)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # plt.figure(figsize=(12, 8))
    # plt.hist(label_distribution, stacked=True,
    #          bins=np.arange(-0.5, num_clients + 1.5, 1),
    #          label=classes, rwidth=0.5)
    # plt.xticks(np.arange(num_clients), ["Client %d" %
    #                                   c_id for c_id in range(num_clients)],rotation=45,fontsize=12)
    # plt.xlabel("客户端 ID",fontsize=14)
    # plt.ylabel("样本数",fontsize=14)
    # plt.legend()
    # plt.title("每个客户端上的标签分布（α = 1）",fontsize=14)
    # # 调整图表布局，确保标签不会重叠
    # plt.tight_layout()
    # plt.show()

    # 客户端分布
    client_dist = [[] for _ in range(num_clients)]

    for client_id, label_num in enumerate(statistic):
        for label in label_num:
            for i in range(label[1]):
                client_dist[client_id].append(label[0])

    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab20', num_clients)
    plt.hist([id for id in client_dist], stacked=True,
             bins=np.arange(-0.5, num_classes + 1.5, 1),
             label=["Client {}".format(i) for i in range(num_clients)], color=[colors(i) for i in range(num_clients)],
             rwidth=0.5)
    plt.xticks(np.arange(n_classes), classes, rotation=45, fontsize=12)
    plt.xlabel("标签类型", fontsize=14)
    plt.ylabel("样本数", fontsize=14)
    plt.legend(loc="upper right")
    plt.title("每个标签在不同客户端上的分布（α = 0.1）", fontsize=14)
    plt.tight_layout()
    plt.show()


    train_data, test_data = split_data(X, y)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
    )


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    np.random.seed(seed)
    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition)
