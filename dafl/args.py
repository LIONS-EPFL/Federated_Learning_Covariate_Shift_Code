from typing import Sequence

import torch
from datargs import argsclass, arg


@argsclass
class Args:
    seed: int = None
    output_dir: str = 'outputs/'
    data_augmentation: bool = True
    dataset: str = 'mnist'
    public_testset_size: int = 100
    num_workers: int = 5

    client_mode: str = 'multi' # 'multi', 'single
    shift: str = 'train'
    train_re: bool = False
    combine_testsets: bool = False
    use_true_ratio :bool = False
    num_classes: int = 10
    
    # Subsampling
    num_client_duplicates: int = 1
    num_subsample: int = -1

    device: str = 'cpu' # Literal['cuda', 'cpu']
    log_interval: int = 100
    test_interval: int = 1000
    
    # Model
    model: str = 'lenet'
    force_pos: bool = False

    optimizer: str = 'adam'
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-6
    batch_size: int = 128
    test_batch_size: int = 512
    num_steps: int = 5000
    batch_norm_agg: str = "FedAvg" # Literal['FedAvg', 'FedBN']

    re_optimizer: str = 'adam'
    re_lr: float = 0.001
    re_momentum: float = 0.9
    re_weight_decay: float = 1e-6
    re_lr_milestones: Sequence[int] = arg(default=())
    re_batch_size: int = 128
    re_batch_size_snd: int = None
    re_batch_size_snd_epoch: int = None
    re_epochs: int = 10
    re_upper_bound: float = 10.0
    re_descent_only: bool = False
    d3re_impl: bool = False
    re_type: str = "lsif" # lsif, pu
    re_batch_drop_last: bool = False

    num_clusters: int = 10

    wandb_project: str = 'dfal'
    wandb_entity: str = 'epfl-lions'
    wandb_name: str = 'debug'
    wandb_id: str = None
    wandb_tags: Sequence[str] = None

    force_grayscale: bool = False

    @property
    def dataloader_kwargs(self):
        return {'num_workers': 4, 'pin_memory': True} if self.device =='cuda' else {}
