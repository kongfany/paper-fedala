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
dir_path = "mnist/"

import pdb
# Allocate data to users
def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"  # 记录了生成数据的划分参数，
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # 这里是做一个检查，如果config.json文件已经存在，并且里面的参数与这次要生成的文件，参数都一样的话，就直接return
    # 不用重复下载生成
    #if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
     #   return

    # FIX HTTP Error 403: Forbidden 爬虫联网
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)

    classes = trainset.classes
    n_classes = len(classes)
    # 数据加载器是用来将训练集和测试集分批次地加载到模型中进行训练和测试的工具
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)
    # shuffle洗牌操作是指在每个训练周期开始前，重新随机打乱训练数据的顺序。这可以帮助模型更好地学习数据之间的关系，避免模型只学习特定顺序的数据。

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())  # 看这里，他把trainset和testset的数据拢在一起了
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)  # 拢在一起的测试集+数据集，叫做dataset，张量
    dataset_label = np.array(dataset_label)
    # (Pdb) p dataset_image.shape
    # (70000, 1, 28, 28)
    # (Pdb) p dataset_label.shape
    # (70000,)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    # X:记录了每个client拥有的数据的 原始内容 的下标（图片）
    # y:记录了每个client拥有的数据的  标签 的下标
    # statistic:记录了每个client拥有的数据类型及数量
    # noiid的两个特征，数据类型异构和数量异构
    # 分离，将数据划分到各个客户端

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition)
    # 分片，分为训练集和测试集
    # pfl每个客户端都有测试集，使得训练的数据分布和测试的数据分布一致，有利于pfl的准确率

    # 可视化
    print(classes,n_classes)


    # pdb.set_trace()
    # 类别
    label_distribution = [[] for _ in range(n_classes)]



    for client_id,label_num in enumerate(statistic):
        # print(client_id,label_num)
        for label in label_num:
            for i in range(label[1]):
                label_distribution[label[0]].append(client_id)
            #print("标签1计算完")
        #print("client0计算完毕")

    #print("所有客户端计算完毕")


    # pdb.set_trace()
    # 客户端分布
    client_dist = [[] for _ in range(num_clients)]

    for client_id,label_num in enumerate(statistic):
        for label in label_num:
            for i in range(label[1]):
                client_dist[client_id].append(label[0])

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    # plt.figure(figsize=(16, 8))
    # plt.hist(label_distribution, stacked=True,
    #          bins=np.arange(-0.5, num_clients + 1.5, 1),
    #          label=classes, rwidth=0.5)
    # plt.xticks(np.arange(num_clients), ["Client %d" %
    #                                   c_id for c_id in range(num_clients)],rotation=45,fontsize=12)
    # plt.xlabel("客户端 ID",fontsize=14)
    # plt.ylabel("样本数",fontsize=14)
    # plt.legend()
    # plt.title("每个客户端上的标签分布",fontsize=14)
    # # 调整图表布局，确保标签不会重叠
    # plt.tight_layout()
    # plt.show()

    # 展示不同label划分到不同client的情况
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab20', num_clients)
    plt.hist([id for id in client_dist], stacked=True,
             bins=np.arange(-0.5, num_classes+ 1.5, 1),
             label=["Client {}".format(i) for i in range(num_clients)],color=[colors(i) for i in range(num_clients)],
             rwidth=0.5)
    plt.xticks(np.arange(n_classes), classes,rotation=45,fontsize=12)
    plt.xlabel("标签类型",fontsize=14)
    plt.ylabel("样本数",fontsize=14)
    plt.legend(loc="upper right")
    plt.title("每个标签在不同客户端上的分布",fontsize=14)
    plt.tight_layout()
    plt.show()






    # 划分训练集和测试集
    train_data, test_data = split_data(X, y)



    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False# 从命令行给定参数，第一个参数是noiid的话
    balance = True if sys.argv[2] == "balance" else False  # 指在联邦学习中每个client分到的样本数相同的场景
    partition = sys.argv[3] if sys.argv[3] != "-" else None  # 指定划分策略名字 Dirichlet

    np.random.seed(seed)
    generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition)
