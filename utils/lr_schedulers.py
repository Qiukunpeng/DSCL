import math
import torch.optim as optim
from torchvision.models import AlexNet
import matplotlib.pyplot as plt


__all__ = [
    "EpochWarmUpLR"
]


def adjust_learning_rate(optimizer, lr):
    """
        Adjust learning rate for optimizer
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        lr (float): Learning rate
    """
    if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]["lr"] = lr
    else:
        optimizer.param_groups[0]["lr"] = lr
        for i in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[0]["lr"] = lr


class EpochWarmUpLR(object):
    def __init__(self, args, optimizer):
        """
            Initialize the learning rate scheduler.
        Args:
            args (argparse.Namespace): Arguments
            optimizer (torch.optim.Optimizer): Optimizer
        """
        self.args = args
        self.optimizer = optimizer

    def __call__(self, warmup_mode, main_mode):
        """
            Choose the warmup mode and main mode of learning rate scheduler.
        Args:
            warmup_mode (str): Warmup mode
            main_mode (str): Main mode

        Returns:
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        """
        if warmup_mode == "none":
            warmup_lr_scheduler = None
        elif warmup_mode == "linear":
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=1/self.args.warmup_epochs, total_iters=self.args.warmup_epochs)
        elif warmup_mode == "constant":
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer=self.optimizer, factor=self.args.factor, total_iters=self.args.warmup_epochs)
        else:
            raise ValueError("Warm mode must be linear or constant!")

        if main_mode == "cosine":
            main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, 
                                                                     T_max=(self.args.epochs - self.args.warmup_epochs), 
                                                                     eta_min=self.args.min_lr)
        elif main_mode == "step":
            main_lr_scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, 
                                                          step_size=self.args.lr_step, 
                                                          gamma=self.args.lr_gamma)
        elif main_mode == "multistep":
            main_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                               milestones=self.args.lr_milestones,
                                                               gamma=self.args.lr_gamma,
                                                               last_epoch=self.args.last_epoch)
        elif main_mode == "exponential":
            main_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, 
                                                                 gamma=self.args.lr_gamma)
        else:
            raise ValueError("LR mode must be cosine, step or exponential!")
        
        if warmup_lr_scheduler is None:
            return main_lr_scheduler
        else:
            return optim.lr_scheduler.SequentialLR(optimizer=self.optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[self.args.warmup_epochs])
