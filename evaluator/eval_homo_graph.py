import numpy as np
import torch
from tqdm import tqdm


from .evaluator import Evaluator
from data import GraphDataset, TCGACancerStageDataset, TCGACancerTypingDataset
from utils import metrics
from parser import parse_gnn_model


class HomoGraphEvaluator(Evaluator):
    def __init__(self, config, verbose=True):
        super().__init__(config, verbose)

        # Initialize GNN model and optimizer
        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)

        # Load trained checkpoint
        state_dict = self.checkpoint_manager.load_model()
        self.gnn.load_state_dict(state_dict)
        self.gnn.eval()
        # Load testing data
        test_path = self.config_data["eval_path"]
        self.name = self.config_data["dataset"]
        self.normal_path = self.config_data["normal_path"] if (self.name == "COAD" or self.name == "BRCA" or self.name== "ESCA") else ""
        self.test_data = self.load_data(test_path)

    def load_data(self, path):
        if self.name == "COAD" or self.name == "BRCA" or self.name == "ESCA":
            type = self.config_data["task"]
            if type == "cancer staging":
                self.average = "macro"
                test_data = TCGACancerStageDataset(path, self.normal_path, "eval")
            elif type == "cancer classification":
                self.average = "macro"
                test_data = GraphDataset(path, self.normal_path, self.name, 'eval')
            elif type == "cancer typing":
                self.average = "binary"
                test_data = TCGACancerTypingDataset(path, self.normal_path, self.name, 'eval')
            else:
                raise ValueError("This task not supported")
        else:
            self.average = "binary"
            test_data = GraphDataset(path, self.normal_path, self.name, 'eval')

        return test_data


    def test_one_step(self, g, label, total, correct):
        g = g.to(self.device)
        with torch.no_grad():
            out = self.gnn(g)
            prob = F.softmax(out)
            pred = out.detach().cpu().numpy().argmax(axis=1)[0]
            prob = prob.detach().cpu().numpy()
        correct += 1 if pred == label else 0
        total += 1
        return total, correct, pred, prob, label

    def eval(self):
        # Initialize metrics
        correct = 0
        total = 0

        if self.verbose:
            testing_range = tqdm(range(len(self.test_data)))
        else:
            testing_range = range(len(self.test_data))
        metrics_log = tqdm(total=0, position=0, bar_format='{desc}')

        pred_list = []
        label_list = []
        prob_list = []
        for idx in testing_range:

            graph, label = self.test_data[idx]

            total, correct, pred, prob, label = self.test_one_step(graph, label, total, correct)
            pred_list.append(pred)
            label_list.append(label)
            prob_list.append(prob)
            if self.verbose:
                testing_range.set_description("Index %d | accuracy: %f" % (idx, correct / total))

        pred_list = np.array(pred_list)
        label_list = np.array(label_list)
        prob_list = np.concatenate(prob_list)

        precision, recall, f1_score, auc = metrics(prob_list, label_list, average=self.average)
        metrics_list = (f1_score, precision, recall, auc)

        if self.verbose:
            metrics_log.set_description_str("Metrics ==> [F1 score: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | AUC: {:.4f}]".format(*metrics_list))

        return correct / total, f1_score, precision, recall, auc