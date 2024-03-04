import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn


def set_seed(args, seed):
    """
        Set seed for all random number generators
    Args:
        args (argparse.Namespace): Arguments
    """
    # For hash
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # For cpu and gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For cudnn
    cudnn.enabled = True
    if args.cuda_deterministic:  # Slower but more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA >= 10.2版本会提示设置这个环境变量
        # torch.use_deterministic_algorithms(True)
    else:  # Faster but less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def worker_seed_init_fn(worker_id, num_workers, local_rank, seed):
    # worker_seed_init_fn function will be called at the beginning of each epoch
    # For each epoch the same worker has same seed value,so we add the current time to the seed
    worker_seed = num_workers * local_rank + worker_id + seed + int(time.time())
    np.random.seed(worker_seed)
    random.seed(worker_seed)
