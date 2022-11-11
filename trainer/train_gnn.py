from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F

from dgl.dataloading import GraphDataLoader

from trainer import Trainer
from evaluator import HomoGraphEvaluator
from parser import parse_optimizer, parse_gnn_model, parse_loss
from utils import acc, metrics
from data import GraphDataset, TCGACancerStageDataset, TCGACancerTypingDataset



class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict):
        super().__init__(config)

        # Initialize GNN model and optimizer
        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)

        # Parse loss function
        self.loss_fcn = parse_loss(self.config_train)

        train_path = self.config_data["train_path"]
        self.valid_path = self.config_data["valid_path"]

        name = self.config_data["dataset"]
        normal_path = self.config_data["normal_path"] if (name == "COAD" or name == "BRCA" or name == "ESCA") else ""
        task = self.config_data["task"]
        if name == "COAD" and task == "cancer staging":
            self.average = "macro"
            train_data = TCGACancerStageDataset(train_path, normal_path, 'train')
        elif name == "BRCA" and task == "cancer staging":
            self.average = "macro"
            train_data = TCGACancerStageDataset(train_path, normal_path, 'train')
        elif (name == "BRCA" or name == "ESCA") and task == "cancer typing":
            self.average = "binary"
            train_data = TCGACancerTypingDataset(train_path, normal_path, 'train')
        else:
            self.average = "binary"
            train_data = GraphDataset(train_path, normal_path, name, 'train')

        self.dataloader = GraphDataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    def train_one_step(self, graphs, label):
        self.optimizer.zero_grad()
        label = label.to(self.device)

        if type(graphs) == tuple:  # Heterogeneous graph
            graphs = [g.to(self.device) for g in graphs]
            pred = [self.gnn(g) for g in graphs]
            pred = torch.cat(pred)
        else:
            graphs = graphs.to(self.device)
            pred = self.gnn(graphs)

        prob = F.softmax(pred)
        loss = self.loss_fcn(pred, label)

        loss.backward()
        self.optimizer.step()

        accuracy = acc(pred, label)

        pred = pred.detach().cpu().numpy().argmax(axis=1)
        prob = prob.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        return loss.item(), accuracy, pred, prob, label

    def train(self) -> None:
        print(f"Start training Homogeneous GNN")

        training_range = tqdm(range(self.n_epoch), nrows=3)
        metrics_log = tqdm(total=0, position=1, bar_format='{desc}')

        for epoch in training_range:
            self.gnn.train()

            res = 0
            pred_list = []
            prob_list = []
            label_list = []
            accuracy_list = []

            for graphs, label in self.dataloader:
                loss, accuracy, pred, prob, label = self.train_one_step(graphs, label)
                res += loss
                accuracy_list.append(accuracy)
                pred_list.append(pred)
                prob_list.append(prob)
                label_list.append(label)

            accuracy = np.mean(accuracy_list)
            pred_list = np.concatenate(pred_list)
            prob_list = np.concatenate(prob_list)
            label_list = np.concatenate(label_list)
            precision, recall, f1_score, train_auc = metrics(prob_list, label_list, average=self.average)

            # Perform validation and testing
            self.checkpoint_manager.save_model(self.gnn.state_dict())
            evaluator = HomoGraphEvaluator(self.config, verbose=False)
            test_acc, test_f1, test_prec, test_recall, test_auc = evaluator.eval()
            evaluator.test_data = evaluator.load_data(self.valid_path)
            val_acc, val_f1, val_prec, val_recall, val_auc = evaluator.eval()

            training_range.set_description_str("Epoch {} | loss: {:.4f}".format(epoch, res))
            metrics_list = (accuracy, f1_score, precision, recall, train_auc,
                            val_acc, val_f1, val_prec, val_recall, val_auc,
                            test_acc, test_f1, test_prec, test_recall, test_auc)
            metrics_log.set_description_str(
                "Metrics ==> [Acc: {:.4f} | F1: {:.4f} | Ps: {:.4f} | Rec: {:.4f} | AUC: {:.4f} |"
                " Val Acc: {:.4f} | Val F1: {:.4f} | Val Ps: {:.4f} | Val Rec: {:.4f} | Val AUC: {:.4f} |"
                " Test Acc: {:.4f} | Test F1: {:.4f} | Test Ps: {:.4f} | Test Rec: {:.4f} | Test AUC: {:.4f}]".format(*metrics_list)
            )

            epoch_stats = {
                "Epoch": epoch + 1,
                "Train Loss: ": res,
                "Training Accuracy": accuracy,
                "Training Precision": precision,
                "Training Recall": recall,
                "Training F1": f1_score,
                "Training AUC": train_auc,
                "Validation Accuracy": val_acc,
                "Validation F1": val_f1,
                "Validation Precision": val_prec,
                "Validation Recall": val_recall,
                "Validation AUC": val_auc,
                "Testing Accuracy": test_acc,
                "Testing F1": test_f1,
                "Testing Precision": test_prec,
                "Testing Recall": test_recall,
                "Testing AUC": test_auc
            }

            # State dict of the model including embeddings
            self.checkpoint_manager.write_new_version(
                self.config,
                self.gnn.state_dict(),
                epoch_stats
            )

            # Remove previous checkpoint
            self.checkpoint_manager.remove_old_version()
