import copy
import numpy as np
import torch
import time
from flcore.clients.clientALA import *
from utils.data_utils import read_client_data
from threading import Thread


class FedALA(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        # 微调参数
        # self.monitor_hessian = args.monitor_hessian

        # 实例化clientala
        self.set_clients(args, clientALA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # 发送模型时初始化了ALA
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(ood_eval=True)

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        print("\nEvaluating Post-Fine-Tuning and OOD Performance...")
        self.evaluate(ood_eval=True)

        # ================= new function =================
        # compute sharpness计算模型的锐度可以帮助评估模型的稳定性和对参数扰动的敏感度。
        # if self.monitor_hessian:
        #     print("Computing sharpness")
        #     eigenvals_list = []
        #     for client in self.select_clients():
        #         eigenvals, _ = client.compute_hessian_eigen()
        #         eigenvals_list.append(eigenvals[0])
        #     print("\nHessian eigenval list: ", eigenvals_list)
        #     print("\nHessian eigenval mean: ", np.mean(eigenvals_list))
        #
        # if self.test_time_adaptation:
        #     self.tta_eval()
        #
        # # save each client model item
        # for client in self.select_clients():
        #     client.save_item(
        #         item=client.model,
        #         item_name=self.goal,
        #         item_path="models/" + self.dataset + "/",
        #     )
        #
        # # ================= new function =================
        #
        # print("\nFine-Tuning with the Last Trained Model......")
        # # add post fine-tuning
        # for client in self.selected_clients:
        #     print("Client LR: ", client.learning_rate)
        #     client.batch_size = 128
        #     client.fixed_soup_flat = True
        # for ft_idx in range(15):
        #     for client in self.selected_clients:
        #         client.train()
        #
        #     # threads = [Thread(target=client.train) for client in self.selected_clients]
        #     # [t.start() for t in threads]
        #     # [t.join() for t in threads]
        #
        #     print("\nFine-Tuning Iteration: ", ft_idx)
        #     self.evaluate(ood_eval=True)


    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.local_initialization(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            print(f'Client {c.id}: Train loss: {cl*1.0/ns}')
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None, ood_eval=False):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        local_test_acc = test_acc
        local_test_auc = test_auc
        if ood_eval :#true
            print("OOD eval details: ")
            # model_ids, num_samples, tot_correct, tot_auc = self.test_metrics(ood_eval=True)

            # id_tot_correct_list = tot_correct[:self.num_clients]
            # id_tot_auc_list = tot_auc[:self.num_clients]
            # id_num_samples_list = num_samples[:self.num_clients]

            # ood_tot_correct_list = tot_correct[self.num_clients:]
            # ood_tot_auc_list = tot_auc[self.num_clients:]
            # ood_num_samples_list = num_samples[self.num_clients:]

            stats_all = self.test_metrics(ood_eval=True)
            stats = stats_all[:4]

            print(stats)
            print(stats[0])

            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
            test_auc = sum(stats[3]) * 1.0 / sum(stats[1])

            # cmh: added per client performance
            test_acc_list = []
            test_auc_list = []

            for idx, c_id in enumerate(stats[0]):
                test_acc_list.append(stats[2][idx] / stats[1][idx])
                test_auc_list.append(stats[3][idx] / stats[1][idx])

            for idx, c_id in enumerate(stats[0]):
                print(
                    "Client {} Test Accurancy: {:.4f}".format(c_id, test_acc_list[idx])
                )
                print("Client {} Test AUC: {:.4f}".format(c_id, test_auc_list[idx]))

            # train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
            accs = [a / n for a, n in zip(stats[2], stats[1])]
            aucs = [a / n for a, n in zip(stats[3], stats[1])]

            print("OOD Performance (client model i on all other dataset j):")
            print("OOD Client-Num-Weighted Test Accurancy: {:.4f}".format(test_acc))
            print("OOD Client-Num-Weighted Test AUC: {:.4f}".format(test_auc))

            print("OOD Client-Equally Test Accurancy: {:.4f}".format(np.mean(accs)))
            print("OOD Client-Equally Test AUC: {:.4f}".format(np.mean(aucs)))

            print("OOD Std Test Accurancy: {:.4f}".format(np.std(accs)))
            print("OOD Std Test AUC: {:.4f}".format(np.std(aucs)))

            print("Performance Summarizing...")
            print("Local Performance:")
            print("Local Client-Equally Test Accurancy: {:.4f}".format(local_test_acc))
            print("Local Client-Equally Test AUC: {:.4f}".format(local_test_auc))

            print("Global Performance:")
            accs.append(local_test_acc)
            aucs.append(local_test_auc)
            print("Glocal Client-Equally Test Accurancy: {:.4f}".format(np.mean(accs)))
            print("Glocal Client-Equally Test AUC: {:.4f}".format(np.mean(aucs)))

            # out-of-federated performance 常用于描述模型在非联邦学习设置下的性能，例如在集中式学习或单个数据源的情况下的性能表现
            if self.hold_out_id != 1e8:
                stats = stats_all[4:]
                print(stats)
                print(stats[0])

                test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
                test_auc = sum(stats[3]) * 1.0 / sum(stats[1])

                print("=" * 16)
                print("OOF Performance:")
                print("OOF Client Test Accurancy: {:.4f}".format(test_acc))
                print("OOF Client Test AUC: {:.4f}".format(test_auc))

