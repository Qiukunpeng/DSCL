import numpy as np
import matplotlib.pyplot as plt

from dataset import MyDataset

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler


class MyDataLoader(object):
    def __init__(self, args, root, logger):
        """
            Initialize the dataloader
        Args:
            args (argparse.Namespace): Arguments
            root (str): The path of the dataset
            logger (logging.Logger): Logger
        """
        self.args = args
        self.root = root
        self.logger = logger

    def get_dataloader(self, split, shuffle):  # Get the dataloader
        """
            Get the dataloader
        Args
            split (str): The type of the dataset
            shuffle (bool optional): Whether to shuffle the dataset

        Returns:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        dataset = MyDataset(args=self.args, root=self.root, logger=self.logger, split=split, transform=True)  # Create the dataset
        if split == "train":
            train_sampler = RandomSampler(data_source=dataset)

            dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.args.batch_size,
                                    sampler=train_sampler,
                                    num_workers=self.args.num_workers,
                                    pin_memory=self.args.pin_memory,
                                    drop_last=self.args.drop_last)  # Create the dataloader

        else:
            test_sampler = SequentialSampler(data_source=dataset)

            dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.args.batch_size,
                                    sampler=test_sampler,
                                    num_workers=self.args.num_workers,
                                    pin_memory=self.args.pin_memory,
                                    drop_last=self.args.drop_last)  # Create the dataloader

        return dataloader  # Return dataloader
