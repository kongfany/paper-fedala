import os
import pdb

import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
train_size = 0.75  # merge original training set and test set, then split it manually.
least_samples = 1  # guarantee that each client must have at least one samples for testing.
alpha = 0.1  # for Dirichlet distribution
# alpha = 1.0

def check(config_path, train_path, test_path, num_clients, num_classes, niid=False,
          balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['num_classes'] == num_classes and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


"""separate_data函数的作用是将数据分割成多个客户端。
首先，它会创建num_clients个空列表X和y，用于存储每个客户端的数据。
还会创建num_clients个空列表statistic，用于存储每个客户端的标签统计信息。

如果partition为'pat'，则按照类别将数据分配给不同的客户端。
首先，根据类别将数据索引分组。
然后，根据class_per_client和balance的值选择一定数量的客户端。
接下来，计算每个客户端应分配的样本数量。
如果balance为True，则将每个客户端的样本数量设置为相等的值；
否则，将每个客户端的样本数量设置为一个随机值。
最后，根据计算出的样本数量将数据分配给每个客户端。 

如果partition为'dir'，则按照Dirichlet分布将数据分配给不同的客户端。
具体的分配过程可以参考给出的链接。
最后，将分配的结果保存在dataidx_map中。 

最后，根据dataidx_map将数据分配给每个客户端，并计算每个客户端的标签统计信息。
最后输出每个客户端的数据大小和标签信息。 
"""


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2):
    # data:二元组，20client，10classes iid，balance，每个客户端默认给两个种类
    X = [[] for _ in range(num_clients)]
    # (Pdb) p X
    # [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    # (Pdb) p len(X)
    # 20
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    # 二维的list

    dataset_content, dataset_label = data  # 传进来的二元组，数据分为两部分，原始内容 标签
    # (Pdb) p dataset_content.shape
    # (70000, 1, 28, 28)

    dataidx_map = {}  # 数据下标的映射的点的初始化，记录每个client分到哪几种label对应的数据，以及下标

    if not niid:  # iid
        partition = 'pat'
        class_per_client = num_classes  # 客户端包含所有种类,10

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        # 创建数组的函数，range()函数是创建一个从0到len(dataset_label)的整数序列的函数。
        # (Pdb) p idxs
        # array([    0,     1,     2, ..., 69997, 69998, 69999])
        # (Pdb) p len(idxs)
        # 70000
        idx_for_each_class = []  # 每个种类的下标
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])
        # 第二维是十个一维列表的二维列表，把所有dateset_label == i的下标，append到idx_for_each_class里面，
        # dataset_label中的元素进行分类，并将每个类别的元素的下标存储在idx_for_each_class中

        # (Pdb) p idx_for_each_class
        # [array([    1,    21,    34, ..., 69964, 69983, 69993])]
        # [array([    1,    21,    34, ..., 69964, 69983, 69993]), array([    3,     6,     8, ..., 69978, 69984, 69994])]
        # (Pdb) p len(idx_for_each_class)
        # 10

        class_num_per_client = [class_per_client for _ in range(num_clients)]  # 每个client的种类，一样的，class_per_client
        for i in range(num_classes):  # 处理每一个类,10
            selected_clients = []
            for client in range(num_clients):  # 20
                # 对于每个客户端，检查其拥有的特定类别的样本数量是否大于0，如果是，则将该客户端添加到`selected_clients`列表中。最终，`selected_clients`列表将包含符合条件的客户端。
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            # (num_clients/num_classes)*class_per_client
            # = num_clients*(class_per_client/num_classes)
            # 采样率q=(class_per_client/num_classes)每个客户端拥有的种类的比例就是采样的比例
            # 每个客户端2个种类采样率就是0.2，
            # iid，10个种类，每个客户端10个种类，采样率为1.
            # 超过这个数量就截断
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])  # 拿到数字i的个数
            num_selected_clients = len(selected_clients)  # 被采样客户端的数量
            num_per = num_all_samples / num_selected_clients  # 每个客户端能拿到多少个i
            # pdb.set_trace()
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
                # 34.515 0.1 345.15 19
            num_samples.append(num_all_samples - sum(num_samples))
            # 把剩下的samples分到最后一个client中，应对num_per的取整问题，8个数据分到3个客户端，332
            # 确定了每个客户端要分多少个数据

            # 划分数据,对于当前类的每个client划分到的数据的下标
            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):  # 对于每个client
                if client not in dataidx_map.keys():  # 将数据下标
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]  # 当前class当前client分到了数据对应的下标
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    # dataidx_map[i]:表示第i个客户端拥有的下标有哪些
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]  # 真正的数据

        for i in np.unique(y[client]):  # 从给定的数组中提取出唯一元素，对提取的唯一元素进行排序
            # 统计每个client 拿到了多少个数据标签为i
            # 为每个客户端统计其拥有的不同数据标签及其数量
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data  # 回收内存，用于删除对某个对象的引用，释放该对象所占用的内存。
    # gc.collect()



    for client in range(num_clients):
        # p_client_id.append(client)
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)


    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):  # i是client的意思
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)  # train_size训练集的比例0.75

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
