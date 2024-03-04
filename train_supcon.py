import os
import time
import math
import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import models
from dataloader import MyDataLoader

from utils import losses
from utils import lr_schedulers

from utils.saver import Saver
from utils.logger import Logger
from utils.seed import set_seed
from utils.metrics import AverageMeter, accuracy
from utils.checkpoint import Checkpoint
from utils.weight_initialization import weight_init

from torch import autograd


class Trainer(object):
    def __init__(self, args):
        """
            Initialize the trainer
        Args:
            args (argparse.Namespace): Arguments
            datatime (time.struct_time): Datatime
        """
        self.args = args
        self.datatime = time.localtime()

        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_ids
        self.device = torch.device("cuda", 0)
        set_seed(args=self.args, seed=self.args.seed)
        
        self.saver_folder_path = Saver(args=self.args)()
        self.logger = Logger(args=self.args, saver_folder_path=self.saver_folder_path)()
        self.checkpoint = Checkpoint(args=self.args, saver_folder_path=self.saver_folder_path, logger=self.logger)

        self.root = os.path.join(self.args.root, self.args.dataset)
        dataloader = MyDataLoader(args=self.args, root=self.root, logger=self.logger)

        self.train_dataloader = dataloader.get_dataloader(split="train", shuffle=True)
        self.length_train_dataloader = len(self.train_dataloader)
        self.length_train_data = len(self.train_dataloader.dataset)

        self.model = models.__dict__[self.args.model](args=self.args, **{"num_classes": self.args.num_classes, })
        self.logger.info(f"Let's use {self.args.loss} loss!")
        self.criterion_1 = losses.__dict__[self.args.loss](args=self.args)
        self.logger.info(f"Let's use {self.args.loss_2} loss!")
        self.criterion_2 = losses.__dict__[self.args.loss_2](args=self.args)

        param_groups = [{"params": self.model.encoder.task_network.parameters(), "weight_decay": self.args.task_weight_decay, "lr_mult": self.args.lr_mult_task},
                        {"params": self.model.encoder.conv_last_1.parameters(), "weight_decay": self.args.weight_decay, "lr_mult": self.args.lr_mult_conv_last},
                        {"params": self.model.encoder.conv_last_2.parameters(), "weight_decay": self.args.weight_decay, "lr_mult": self.args.lr_mult_conv_last},
                        {"params": self.model.encoder.conv_last_3.parameters(), "weight_decay": self.args.weight_decay, "lr_mult": self.args.lr_mult_conv_last},
                        {"params": self.model.encoder.conv_last_4.parameters(), "weight_decay": self.args.weight_decay, "lr_mult": self.args.lr_mult_conv_last},
                        {"params": self.model.encoder.saliency_network.parameters(), "weight_decay": self.args.weight_decay, "lr_mult": self.args.lr_mult_saliency}]
        
        self.optimizer = torch.optim.SGD(params=param_groups,
                                         lr=self.args.lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay,
                                         nesterov=self.args.nesterov)
        
        scheduler = lr_schedulers.__dict__[self.args.lr_scheduler](args=self.args, optimizer=self.optimizer)
        self.scheduler = scheduler(warmup_mode=self.args.warmup_mode, main_mode=self.args.main_mode)
    
        self.logger.info("Let's use single GPU!")
        self.model = self.model.to(self.device)
        self.criterion_1 = self.criterion_1.to(self.device)
        self.criterion_2 = self.criterion_2.to(self.device)
        self.checkpoint.save_config()
        self.logger.info(self.args)
        
        self.best_loss = 10000.0
            
    def adjust_learning_rate(self, optimizer):

        param_group_0 = optimizer.param_groups[0]

        for i in range(1, len(optimizer.param_groups)):
            param_group = optimizer.param_groups[i]
            param_group["lr"] = param_group["lr_mult"] * param_group_0["lr"]

    def train(self, epoch):
        """
            Train the model
        Args:
            epoch (int): Epoch
        """
        self.logger.info("Training Model...")
        self.model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        tbar = tqdm(self.train_dataloader)
            
        # if epoch > 30:
        #     p = 1
        # else:
        #     p = 0
            
        end = time.time()
        for batch_index, (images, labels) in enumerate(tbar):
            data_time.update(time.time() - end)

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            bsz = labels.shape[0]

            outputs_saliency, outputs_contrastive = self.model(images, 1)
            
            loss_2 = self.criterion_2(outputs_saliency, labels)
                
            f, f1, f2, f3, f4 = torch.split(outputs_contrastive, [bsz, bsz, bsz, bsz, bsz], dim=0)
            outputs_contrastive = torch.cat([f.unsqueeze(1), f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1), f4.unsqueeze(1)], dim=1)
            loss_1 = self.criterion_1(outputs_contrastive, labels)
                
            loss = self.args.lamda_1 * loss_1 + self.args.lamda_2 * loss_2

            losses.update(loss.item(), bsz)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            tbar.set_description(f"Training Epoch: {epoch + 1}"
                                f"\tnumImages: [{batch_index * self.args.batch_size + labels.size(0)}/{self.length_train_data}]"
                                f"\tIter Loss: {losses.val:.4f}"
                                f"\tLR: {self.optimizer.param_groups[0]['lr']:.6f}")

        train_loss = losses.avg
  
        self.logger.info(f"Trained Epoch: [{epoch + 1}/{self.args.epochs}]"
                         f"\tLoss: {train_loss:.4f}"
                         f"\tTime Consumed-Iter: {batch_time.avg:.2f}s"
                         f"\tTime Consumed-Epoch: {batch_time.sum:.2f}s")
        
        if self.args.no_val:
            if train_loss < self.best_loss:
                is_best = True
                self.best_loss = train_loss
                self.checkpoint.save_checkpoint(is_best=is_best, best_accuracy=self.best_loss, epoch=epoch, model=self.model, 
                                                optimizer=self.optimizer, lr_scheduler=self.scheduler)
            else:
                self.logger.info(f"Best Loss is still: {self.best_loss:.4f}"
                                 f"\tNo checkpoint file will be saved!")

def main():
    """
        Main Function
    """
    parser = argparse.ArgumentParser(description="PyTorch Classification Training")
    # Configuration of training process
    # torchrun --nproc_per_node=4 train.py --experiment=0
    parser.add_argument("--gpu_ids", type=str, default="0, 1, 2, 3", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--cuda_deterministic", type=bool, default=True,
                        help="whether to set deterministic options for CUDNN backend.")
    # Configuration of model
    parser.add_argument("--model", type=str, default="sup_con_resnet", help="model name")
    parser.add_argument("--num_classes", type=int, default=3, help="number of classes (default: 3)")
    parser.add_argument("--start_epoch", type=int, default=0, help="manual epoch number (useful on restarts)")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs to train (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for saliency map")
    parser.add_argument("--projector", type=str, default="Linear", choices=["Linear", "MLP_2", "MLP_3"], help="projection of feature embeddings")
    # Configuration of dataloader
    parser.add_argument("--root", type=str, default="./data", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="Fold-0", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training (default: 32)")
    parser.add_argument("--train_size", type=int, default=512, help="size of train dataset (default: 224)")
    parser.add_argument("--val_size", type=int, default=512, help="size of val dataset (default: 224)")
    parser.add_argument("--interpolation", type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic", "lanczos"], 
                        help="interpolation mode (default: bilinear)")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader (default: 0)")
    parser.add_argument("--pin_memory", type=bool, default=True, help="use pin memory or not (default: False)")
    parser.add_argument("--drop_last", type=bool, default=False,
                        help="drop last batch or not when the batch size is not enough (default: False)")
    # Configuration of loss function
    parser.add_argument("--loss", type=str, default="DSCL", choices=["CELoss", "FocalCELoss", "LabelSmoothCELoss", "DSCL"],
                        help="loss function mode (default: CELoss)")
    parser.add_argument("--loss_2", type=str, default="CELoss", choices=["CELoss", "FocalCELoss", "LabelSmoothCELoss", "SupConLoss", "DCL", "DCLW"],
                        help="loss function mode (default: CELoss)")
    parser.add_argument("--lamda_1", type=float, default=1.0, help="lamda is the weight of the loss 1 (default: 0.5)")
    parser.add_argument("--lamda_2", type=float, default=1.0, help="lamda is the weights of the loss 2 (default: 0.5)")
    parser.add_argument("--ignore_index", type=int, default=-1, help="ignore index for loss function (default: -1)")
    parser.add_argument("--reduction", type=str, default="mean", choices=["mean", "sum"], help="reduction mode (default: mean)")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="label smoothing rate (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=2.0, help="gamma for FocalCELoss (default: 2.0)")
    # Configuration of optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"], help="optimizer mode")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate (default: 0.1)")
    parser.add_argument("--lr_mult_task", type=float, default=1, help="rate of task network with base lr")
    parser.add_argument("--lr_mult_conv_last", type=float, default=1e-2, help="rate of convolution layer with base lr")
    parser.add_argument("--lr_mult_saliency", type=float, default=1, help="rate of saliency network with base lr")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum (default: 0.9)")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay (default: 5e-4)")
    parser.add_argument("--task_weight_decay", type=float, default=1e-4, help="weight decay (default: 5e-4)")
    parser.add_argument("--nesterov", type=bool, default=True, help="nesterov (default: True)")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="betas (default: (0.9, 0.999))")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps (default: 1e-8)")
    parser.add_argument("--amsgrad", type=bool, default=False, help="amsgrad (default: False)")
    # Configuration of lr scheduler
    parser.add_argument("--lr_scheduler", type=str, default="EpochWarmUpLR",
                        choices=["CosineLR", "MultiStepLR", "ExponentialLR", "EpochWarmUpLR"],
                        help="learning rate scheduler mode (default: EpochWarmUpLR)")
    parser.add_argument("--warmup_mode", type=str, default="none", choices=["none", "linear", "constant"], 
                        help="warmup mode (default: linear)")
    parser.add_argument("--main_mode", type=str, default="cosine", choices=["cosine", "step", "multistep", "exponential"], 
                        help="main mode (default: cosine)")
    parser.add_argument("--min_lr", type=float, default=0, help="minimum learning rate (default: 1e-6)")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="warm up training phase (default: 5)")
    parser.add_argument("--factor", type=float, default=0.1, help="learning rate decay factor (default: 0.1)")
    parser.add_argument("--lr_step", type=int, default=30,
                        help="step size for step learning rate scheduler, only used when lr_scheduler=step (default: 30)")
    parser.add_argument("--lr_milestones", type=list, default=[120, 160], help="Learning rate milestones (default: [25, 55, 75])")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="learning rate decay factor (default: 0.1)")
    parser.add_argument("--last_epoch", type=int, default=-1, help="Last epoch")
    # Configuration of checkpoint
    parser.add_argument("--saver", type=str, default="savers",
                        help="path to save checkpoint, log and tensorboard (default: savers)")
    parser.add_argument("--experiment", type=int, default=0, help="experiment name")
    parser.add_argument("--log", type=str, default="logs", help="path to save log (default: logs)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints",
                        help="path to save checkpoint (default: checkpoints)")
    # Configuration of validation
    parser.add_argument("--eval_interval", type=int, default=1, help="evaluation interval (default: 1)")
    parser.add_argument("--no_val", type=bool, default=True, help="validation during training")
    # Create the parser
    args = parser.parse_args()

    trainer = Trainer(args)
    
    start = time.time()
    
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        
        trainer.adjust_learning_rate(trainer.optimizer)
        
        trainer.logger.info(f"=> Epoch: {epoch + 1}"
                            f"\tLR: {trainer.optimizer.param_groups[0]['lr']:.8f}"
                            f"\tBest Loss: {trainer.best_loss:.4f}")

        trainer.train(epoch)
        if not args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
        trainer.scheduler.step()

    end = time.time()
    total_time = str(datetime.timedelta(seconds=int(end - start)))
    trainer.logger.info(f"Total Time Consumed: {total_time}")


if __name__ == "__main__":
    main()
