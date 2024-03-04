import torch
import torch.nn as nn


def weight_init(layer):
    """
        Initialize weights of a layer
    Args:
        layer (nn.Module): Layer to initialize weights
    """
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)


