import random

import numpy as np
import qiskit_algorithms
import torch


def fix_seed(seed: int):
    """Fix the random seeds to have reproducibility.

    :param int seed: seed
    """
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    qiskit_algorithms.utils.algorithm_globals.random_seed = seed


def callback_print(weights: np.ndarray, obj_func_eval: float):
    """Print the objective function value.

    :param np.ndarray weights: current weights
    :param float obj_func_eval: objective function value
    """
    print(f"current_value: {obj_func_eval}")
