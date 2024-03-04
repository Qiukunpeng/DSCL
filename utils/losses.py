import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.seed import set_seed


torch.set_printoptions(profile='full')

__all__ = [
    "CELoss",
    "FocalCELoss",
    "LabelSmoothCELoss",
    "DSCL",
]


class CELoss(nn.Module):
    def __init__(self, args):
        """
            Cross Entropy Loss
        Args:
            args (argparse): The arguments
        """
        super(CELoss, self).__init__()
        self.args = args
        self.criteria = nn.CrossEntropyLoss(weight=None,
                                            ignore_index=self.args.ignore_index,
                                            reduction=self.args.reduction,
                                            label_smoothing=self.args.label_smoothing)

    def forward(self, logits, targets):
        """
            Forward pass
        Args:
            logits (torch.Tensor): Logits
            targets (torch.Tensor): Targets

        Returns:
            loss (torch.Tensor): Loss
        """
        loss = self.criteria(logits, targets)
        return loss


class FocalCELoss(nn.Module):
    def __init__(self, args):
        """
            Focal Cross Entropy Loss
        Args:
            args (argparse): The arguments
        """
        super(FocalCELoss, self).__init__()
        self.args = args

    def forward(self, logits, targets):
        """
            Forward pass
        Args:
            logits (torch.Tensor): Logits
            targets (torch.Tensor): Targets

        Returns:
            loss (torch.Tensor): Loss
        """
        softmax_logits = F.softmax(logits, dim=1)
        one_hot_label = F.one_hot(targets, softmax_logits.size(1)).float()
        pt = torch.where(torch.eq(one_hot_label, 1.), softmax_logits, 1. - softmax_logits)
        focal_weight = torch.pow((1. - pt), self.args.gamma)

        loss = (-F.log_softmax(logits, dim=1)) * one_hot_label
        loss = focal_weight * loss
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()
        return loss


class LabelSmoothCELoss(nn.Module):
    def __init__(self, args):
        """
            Label Smooth Cross Entropy Loss
        Args:
            args (argparse): The arguments
        """
        super(LabelSmoothCELoss, self).__init__()
        self.args = args

    def forward(self, logits, targets):
        """
            Forward pass
        Args:
            logits (torch.Tensor): Logits
            targets (torch.Tensor): Targets

        Returns:
            loss (torch.Tensor): Loss
        """
        logits = F.log_softmax(logits, dim=1)
        one_hot_label = F.one_hot(targets, logits.size(1)).float()
        smoothed_one_hot_label = (
            1. - self.args.label_smoothing) * one_hot_label + self.args.label_smoothing / logits.size(1)
        loss = (-logits) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()
        return loss
    
    
class DSCL(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, args, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(DSCL, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # print(f"The shape of mask is {mask.shape}")
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast
        exp_logits = torch.exp(logits)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        
        num_positives_per_row  = torch.sum(positives_mask , axis=1)
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True) - exp_logits * positives_mask
        log_probs = logits - torch.log(denominator)
        
        if torch.any(torch.isnan(log_probs)):
            print(torch.log(denominator))
            print(denominator)
            raise ValueError("Log_prob has nan!")
        
        log_probs = torch.sum(log_probs * positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

        loss = - log_probs
        loss *= (self.temperature / self.base_temperature)
        loss = loss.mean()

        return loss
