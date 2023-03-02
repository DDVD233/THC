from util import Logger, accuracy, TotalMeter, topk_dominate_loss
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data
from util.loss import mixup_cluster_loss
import random
from models import THC
import wandb
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class BasicTrain:

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizers = optimizers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.group_loss = train_config['group_loss']
        # self.group_loss_weight = train_config['group_loss_weight']

        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']

        self.dominate_loss = train_config['dominate_loss'] if "dominate_loss" in train_config else None
  
        self.dominate_loss_weight = train_config['dominate_loss_weight'] if "dominate_loss_weight" in train_config else None

        # self.dominate_softmax = train_config['dominate_softmax']

        self.topk = train_config['topk'] if "topk" in train_config else None

        # self.save_path = Path(f"{train_config['log_folder']}/{}_{}")
        self.save_path = log_folder

        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy, self.edges_num = [
                TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def train_per_epoch(self, optimizer):
        self.model.train()

        for data_in, pearson, label in self.train_dataloader:
            label = label.long()

            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)

            inputs, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)

            output, learnable_matrix, edge_variance = self.model(inputs, nodes)

            loss = 2 * mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            if self.group_loss:
                loss += mixup_cluster_loss(learnable_matrix,
                                           targets_a, targets_b, lam)

            if self.sparsity_loss:
                sparsity_loss = self.sparsity_loss_weight * \
                    torch.norm(learnable_matrix, p=1)
                loss += sparsity_loss
            if self.dominate_loss:
                dominate_graph_ls = self.dominate_loss_weight * \
                    topk_dominate_loss(learnable_matrix, k=self.topk)
                # print(dominate_graph_ls.item())
                loss += dominate_graph_ls

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            self.edges_num.update_with_weight(edge_variance, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for data_in, pearson, label in dataloader:
            label = label.long()
            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)
            output, _, _ = self.model(data_in, pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        return [auc] + list(metric)

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []

        for data_in, nodes, label in self.test_dataloader:
            label = label.long()
            data_in, nodes, label = data_in.to(
                device), nodes.to(device), label.to(device)
            _, learable_matrix, _ = self.model(data_in, nodes)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}'
            ]))

            wandb.log({
                "Epoch": epoch, 
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Val AUC": val_result[0],
                "Test AUC": test_result[0]
            })
            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result + test_result)

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)


class BrainGNNTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super(BrainGNNTrain, self).__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False
        self.diff_loss = train_config.get('diff_loss', False)
        self.cluster_loss = train_config.get('cluster_loss', True)
        self.assignment_loss = train_config.get('assignment_loss', True)

    def train_per_epoch(self, optimizer):

        self.model.train()

        for data_in, pearson, label in self.train_dataloader:
            label = label.long()

            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)

            _, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)


            output, assignments = self.model(nodes)
            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)
            if self.cluster_loss or self.assignment_loss:
                additional_loss = self.model.loss(assignments)
                if additional_loss is not None:
                    loss += additional_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for data_in, pearson, label in dataloader:
            label = label.long()

            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)

            output, assignments = self.model(pearson)
            # x = torch.reshape(x, (data.num_graphs, -1, x.shape[-1]))

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        return [auc] + list(metric)


