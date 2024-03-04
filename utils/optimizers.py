import torch


__all__ = [
    "sgd",
    "adam",
    "adamw",
]


def decay_or_not(model):
    """
        Filter out the parameters that do not need to be decayed.
    Args:
        model (nn.Module): model

    Returns:
        param_groups (list): list of dict
    """
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == "bias")
    others_list = (param for name, param in model.named_parameters() if name[-4:] != "bias")
    
    param_groups = [{"params": no_decay_list, "weight_decay": 0},
                    {"params": others_list},]
    
    return param_groups


def sgd(args, model):
    """
        SGD optimizer
    Args:
        args (argparse): The arguments
        model (nn.Module): model

    Returns:
        optimizer (torch.optim): optimizer
    """
    param_groups = decay_or_not(model)
    optimizer = torch.optim.SGD(params=param_groups,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    return optimizer


def adam(args, model):
    """
        Adam optimizer
    Args:
        args (argparse): The arguments
        model (nn.Module): model

    Returns:
        optimizer (torch.optim): optimizer
    """
    param_groups = decay_or_not(model)
    optimizer = torch.optim.Adam(params=param_groups,
                                 lr=args.lr,
                                 betas=args.betas,
                                 eps=args.eps,
                                 weight_decay=args.weight_decay,
                                 amsgrad=args.amsgrad)

    return optimizer


def adamw(args, model):
    """
        AdamW optimizer
    Args:
        args (argparse): The arguments
        model (nn.Module): model

    Returns:
        optimizer (torch.optim): optimizer
    """
    param_groups = decay_or_not(model)
    optimizer = torch.optim.AdamW(params=param_groups,
                                  lr=args.lr,
                                  betas=args.betas,
                                  eps=args.eps,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad)

    return optimizer

