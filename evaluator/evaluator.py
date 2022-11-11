from abc import ABC
from checkpoint import CheckpointManager


class Evaluator(ABC):
    def __init__(self, config, verbose=True) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["datasets"]
        self.config_train = config['train']
        self.config_eval = config['eval']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoint']
        self.config_gnn = config["GNN"]

        self.verbose = verbose

        # GPU configs
        self.device = "cuda" if config['gpu_ids'] else "cpu"

        # Define checkpoint manager
        self.checkpoint_manager = CheckpointManager(path=config['checkpoint']["path"])

        if verbose:
            print(
                f"Loaded checkpoint with path {config['checkpoint']['path']} version {self.checkpoint_manager.version}"
            )
