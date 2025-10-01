import pdb

import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple

class ALA:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple], 
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 10) -> None:
        """
        Initialize ALA module

        Args:
            cid: Client ID. 
            loss: The loss function. 
            train_data: The reference of the local training data.训练集
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None # Learnable local aggregation weights.可学习的局部聚合权值。
        self.start_phase = True


    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.

        adaptive_local_aggregation：在权重学习之前准备工作，随机选择局部训练数据并保留更新的较低层。
        由于模型中的参数将在 ALA 过程中发生变化，因此为了方便起见，我们使用它们的引用。
        然后，它学习本地聚合的权重。
        首先，为了避免影响局部模型训练，我们将局部模型克隆为临时局部模型，仅用于在权重学习过程中获得局部模型反向传播后的梯度;
        然后，我们冻结了临时局部模型下层的参数，以防止 Pytorch 中的梯度计算。在权重训练之前，我们初始化权重和温度局部模型。
        之后，我们在第二次迭代中训练权重直到收敛，并在后续迭代中只训练一个 epoch。
        最后，我们将临时局部模型的参数设置为局部模型的相应参数，得到初始化后的局部模型。

        这段代码的主要目的是实现自适应本地聚合 (ALA) 算法中的权重学习过程。以下是该代码的主要步骤：

        初始化权重并创建临时本地模型以进行权重学习。
        冻结低层参数，以减少在 PyTorch 中的计算成本。
        使用随机采样的部分本地训练数据创建数据加载器。
        在每个批次中，计算输出并计算损失值，然后根据损失值更新权重。
        记录损失值，并在达到一定标准差以下时停止权重训练。
        将初始化的本地模型参数更新为学习后的参数。
        这段代码的目的是在分布式学习中实现本地模型的自适应聚合，以便在保持模型准确性的同时减少通信量和计算成本。

        """

        # randomly sample partial local training data 随机抽取局部训练数据
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio*len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data)-rand_num)
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx+rand_num], self.batch_size, drop_last=True)


        # obtain the references of the parameters 获取各参数的参考信息
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration 在第一次通信迭代时停用ALA 全局模型参数和客户端模型参数一致
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers 保留较低层的所有更新 全局更新的低层
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()


        # temp local model only for weight learning
        # 创建了本地模型的一个临时副本 `model_t`，并提取了该临时模型的参数列表存储在 `params_t` 中
        # 这样可以在不影响原始本地模型的情况下进行一些权重学习的操作。
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()# 梯度归零
                output = model_t(x)
                loss_value = self.loss(output, y) # modify according to the local objective
                loss_value.backward()# 反向传播

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            #print(losses)
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                # print('Client:', self.cid, '\tALA epochs:', cnt)
                break

            # train the weight until convergence
            # (Pdb) p self.num_pre_loss
            # 10
            # (Pdb) p len(losses)
            # 11
            # (Pdb) np.std(losses[-self.num_pre_loss:])
            # 0.05621176765198111
            # (Pdb) p self.threshold
            # 0.1
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                    '\tALA epochs:', cnt)
                break


        # 跳出循环
        # pdb.set_trace()
        print(losses)
        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()