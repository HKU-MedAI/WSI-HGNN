from abc import ABC
from collections import OrderedDict

from checkpoint import CheckpointManager


class Trainer(ABC):
    def __init__(self, config: OrderedDict) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["datasets"]
        self.config_train = config['train']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoint']
        self.config_gnn = config["GNN"]

        # Read name from configs
        self.name = config['name']

        # Define checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.config_checkpoint['path'])
        self.save_steps = self.config_checkpoint["save_checkpoint_freq"]

        # Reading number of epochs
        self.n_epoch = self.config_train['num_epochs']

        # Number of workers and batch size
        self.num_workers = self.config_data["num_workers"]
        self.batch_size = self.config_train['batch_size']

        # Load device for training
        self.gpu_ids = config['gpu_ids']
        self.device = "cuda" if config['gpu_ids'] else "cpu"
        self.use_gpu = True if self.device == "cuda" else False

    def train(self) -> None:
        raise NotImplementedError
