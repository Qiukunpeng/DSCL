import os
import torch
from collections import OrderedDict


class Checkpoint(object):
    def __init__(self, args, saver_folder_path, logger):
        """
            Initialize checkpoint
        Args:
            args (argparse.Namespace): Arguments
            saver_folder_path (str): Path to save checkpoint
            logger (logging.Logger): Logger
        """
        self.args = args
        self.saver_folder_path = saver_folder_path
        self.logger = logger

        self.checkpoint_folder_path = os.path.join(self.saver_folder_path, self.args.checkpoint)
        if not os.path.exists(self.checkpoint_folder_path):
            os.makedirs(self.checkpoint_folder_path, exist_ok=True)

    
    def save_checkpoint(self, is_best, best_accuracy, epoch, model, optimizer, lr_scheduler):
        """
            Save checkpoint
        Args:
            is_best (bool): Whether current checkpoint is the best
            best_accuracy (float): The best accuracy of testing
            epoch (int): Current epoch
            model (torch.nn.Module): Model
            optimizer (torch.optim.Optimizer): Optimizer
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        """
        if is_best:
            if self.args.loss == "SupConLoss":
                with open(os.path.join(self.checkpoint_folder_path, "best_loss.txt"), "w") as f:
                    f.write(f"Epoch: {epoch + 1}")
                    f.write(f"\nBest Loss: {best_accuracy:.4f}")
            else:
                with open(os.path.join(self.checkpoint_folder_path, "best_accuracy.txt"), "w") as f:
                    f.write(f"Epoch: {epoch + 1}")
                    f.write(f"\nBest Accuracy: {100 * best_accuracy:.2f}%")

            checkpoint_file_path = os.path.join(self.checkpoint_folder_path,
                                                f"{self.args.dataset.split('/')[-1]}-{self.args.model}-best-{epoch + 1}.pth")

            state_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "epoch_state_dict": epoch,
                "best_accuracy_state_dict": best_accuracy
            }
                
            if self.args.loss == "SupConLoss":
                self.logger.info(f"Best Loss is updated: {best_accuracy:.4f}"
                                 f"\tSaving checkpoint file to {checkpoint_file_path}!")
            else:
                self.logger.info(f"Best Accuracy is updated: {100 * best_accuracy:.2f}%"
                                f"\tSaving checkpoint file to {checkpoint_file_path}!")

            torch.save(state_dict, checkpoint_file_path)

    def load_checkpoint(self, model, optimizer, lr_scheduler, model_ema, scaler):
        """
            Load checkpoint
        Args:
            model (nn.Module): Model
            optimizer (torch.optim): Optimizer
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
            model_ema (torch.nn.Module): Model with exponential moving average
            scaler (torch.cuda.amp.GradScaler): Grad scaler

        Returns:
            model (nn.Module): Model: Model
            optimizer (torch.optim): Optimizer
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
            best_accuracy (float): The best accuracy of testing
            start_epoch (int): Start epoch
            model_ema (nn.Module): Model with exponential moving average
            scaler (torch.cuda.amp.GradScaler): Grad scaler
        """
        best_checkpoint_file = [f for f in os.listdir(self.checkpoint_folder_path) if f.endswith(".pth")][-1]

        if best_checkpoint_file:
            checkpoint_file_path = os.path.join(self.checkpoint_folder_path, best_checkpoint_file)

            self.logger.info(f"=> Found best accuracy checkpoint file: {checkpoint_file_path}")
            self.logger.info("=> Loading checkpoint file...")

            checkpoint = torch.load(checkpoint_file_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            interrupted_epoch = checkpoint["epoch_state_dict"]
            best_accuracy = checkpoint["best_accuracy_state_dict"]
            
            start_epoch = interrupted_epoch + 1
            
            if self.args.model_ema:
                model_ema.load_state_dict(checkpoint["model_ema_state_dict"])
            else:
                model_ema = None
                
            if self.args.amp:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            else:
                scaler = None
                
            return model, optimizer, lr_scheduler, best_accuracy, start_epoch, model_ema, scaler

        else:
            raise FileNotFoundError("=> No checkpoint file found!")

    def save_config(self):
        """
            Save config file
        """
        experiment_config_file_path = os.path.join(self.checkpoint_folder_path, "config.txt")
        
        if self.args.model == "sup_con_resnet":

            with open(experiment_config_file_path, "w") as f:
                p = OrderedDict()
                # Configuration of training process
                p["Seed"] = self.args.seed
                p["GPUs"] = self.args.gpu_ids
                # Configuration of models
                p["Dataset"] = self.args.dataset
                p["Model"] = self.args.model
                p["Epochs"] = self.args.epochs
                p["Temperature for Saliency Map"] = self.args.temperature
                p["Projection of Feature Embeddings"] = self.args.projector
                # Configuration of dataloader
                p["Batch Size"] = self.args.batch_size
                p["Train Size"] = self.args.train_size
                p["Val Size"] = self.args.val_size
                p["Interpolation"] = self.args.interpolation
                # Configuration of loss function
                p["Loss Function of Contrastive"] = self.args.loss
                p["Loss Function of Saliency"] = self.args.loss_2
                p["Weight of the Loss 1"] = self.args.lamda_1
                p["Weight of the Loss 2"] = self.args.lamda_2
                p["Ignore Index of CELoss"] = self.args.ignore_index
                p["Reduction of CELoss"] = self.args.reduction
                p["Label Smoothing of CELoss"] = self.args.label_smoothing
                # Configuration of optimizer
                p["Optimizer"] = self.args.optimizer
                p["Learning Rate"] = self.args.lr
                p["Rate of Task Network with Base LR"] = self.args.lr_mult_task
                p["Rate of Convolution Layer with Base LR"] = self.args.lr_mult_conv_last
                p["Rate of Saliency Network with Base LR"] = self.args.lr_mult_saliency
                p["Momentum of SGD"] = self.args.momentum
                p["Weight Decay"] = self.args.weight_decay
                p["Weight Decay of Task Network"] = self.args.task_weight_decay
                p["Nesterov of SGD"] = self.args.nesterov
                p["Betas of Adam(W)"] = self.args.betas
                p["Epsilon of Adam(W)"] = self.args.eps
                p["Amsgrad of Adam(W)"] = self.args.amsgrad
                # Configuration of lr scheduler
                p["Scheduler"] = self.args.lr_scheduler
                p["Warmup Scheduler Mode"] = self.args.warmup_mode
                p["Main Scheduler Mode"] = self.args.main_mode
                p["Minimum Learning Rate"] = self.args.min_lr
                p["Warmup Epochs"] = self.args.warmup_epochs
                p["Factor"] = self.args.factor
                p["Learning Rate Step"] = self.args.lr_step
                p["Learning Rate Milestones"] = self.args.lr_milestones
                p["Learning Rate Decay Factor "] = self.args.lr_gamma
                
                for key, val in p.items():
                    f.write(key + ": " + str(val) + "\n")
                    
        else:
            with open(experiment_config_file_path, "w") as f:
                p = OrderedDict()
                # Configuration of training process
                p["Seed"] = self.args.seed
                p["GPUs"] = self.args.gpu_ids
                # Configuration of models
                p["Dataset"] = self.args.dataset
                p["Pretrained Model"] = self.args.pretrained_model
                p["Model"] = self.args.model
                p["Epochs"] = self.args.epochs
                p["Temperature for Saliency Map"] = self.args.temperature
                p["Projection of Feature Embeddings"] = self.args.projector
                # Configuration of dataloader
                p["Batch Size"] = self.args.batch_size
                p["Train Size"] = self.args.train_size
                p["Val Size"] = self.args.val_size
                p["Interpolation"] = self.args.interpolation
                # Configuration of loss function
                p["Loss Function"] = self.args.loss
                p["Ignore Index of CELoss"] = self.args.ignore_index
                p["Reduction of CELoss"] = self.args.reduction
                p["Label Smoothing of CELoss"] = self.args.label_smoothing
                # Configuration of optimizer
                p["Optimizer"] = self.args.optimizer
                p["Learning Rate"] = self.args.lr
                p["Rate of Task Network with Base LR"] = self.args.lr_mult_task
                p["Momentum of SGD"] = self.args.momentum
                p["Weight Decay"] = self.args.weight_decay
                p["Nesterov of SGD"] = self.args.nesterov
                p["Betas of Adam(W)"] = self.args.betas
                p["Epsilon of Adam(W)"] = self.args.eps
                p["Amsgrad of Adam(W)"] = self.args.amsgrad
                # Configuration of lr scheduler
                p["Scheduler"] = self.args.lr_scheduler
                p["Warmup Scheduler Mode"] = self.args.warmup_mode
                p["Main Scheduler Mode"] = self.args.main_mode
                p["Minimum Learning Rate"] = self.args.min_lr
                p["Warmup Epochs"] = self.args.warmup_epochs
                p["Factor"] = self.args.factor
                p["Learning Rate Step"] = self.args.lr_step
                p["Learning Rate Milestones"] = self.args.lr_milestones
                p["Learning Rate Decay Factor "] = self.args.lr_gamma
                
                for key, val in p.items():
                    f.write(key + ": " + str(val) + "\n")
