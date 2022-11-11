from globals import *

import yaml
import argparse
import random

import torch

from utils import ordered_yaml
from trainer import GNNTrainer
from evaluator import HomoGraphEvaluator, ExplainGraph

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="")
parser.add_argument('-seed', type=int, help='random seed of the run', default=611)

args = parser.parse_args()

opt_path = args.config
default_config_path = "BRCA/HEAT2_kimia_classification_v2.yml"

if opt_path == "":
    opt_path = CONFIG_DIR / default_config_path

# Set seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)

#############################################################
# Set modes:
# train: initialize trainer for classification
# eval: Evaluate the trained model quantitatively
# construct_graph: Construct graphs from WSI and save to disk
# graph_explain: Explain the GNN and plot the results
#############################################################
mode = "train"

def main():
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    if mode == "train":
        if config["train_type"] == "gnn":
            trainer = GNNTrainer(config)
        else:
            raise NotImplementedError("This type of model is not implemented")
        trainer.train()
    elif mode == "eval":
        if config["eval_type"] == "homo-graph":
            evaluator = HomoGraphEvaluator(config)
        else:
            raise NotImplementedError("This type of evaluator is not implemented")
        evaluator.eval()
    elif mode == "graph_explain":
        explainer = ExplainGraph(config)
        explainer.eval()
    elif mode == "construct_graph":
        pass


if __name__ == "__main__":
    main()


