import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import qiskit_algorithms
import torch

# For callback functions.
global_current_step = 0
global_objective_func_vals = []
GLOBAL_GRAPH_PATH = "./objective_values.png"


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
    global global_current_step
    global_current_step += 1
    print(f"Iter {global_current_step}, current_value: {obj_func_eval}")


def callback_mlflow(weights: np.ndarray, obj_func_eval: float):
    """Save the objective function value as train_loss to mlflow.

    :param np.ndarray weights: current weights
    :param float obj_func_eval: objective function value
    """
    global global_current_step
    global_current_step += 1
    mlflow.log_metric(f"train_loss", obj_func_eval, step=global_current_step)


def callback_graph(weights: np.ndarray, obj_func_eval: float):
    """Save the objective function value as train_loss to mlflow.

    :param np.ndarray weights: current weights
    :param float obj_func_eval: objective function value
    """
    global global_objective_func_vals
    global_objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(global_objective_func_vals)), global_objective_func_vals)

    global GLOBAL_GRAPH_PATH
    plt.savefig(GLOBAL_GRAPH_PATH)


def callback_all(weights: np.ndarray, obj_func_eval: float):
    """Print and save the objective function value to mlflow, and save the fig.

    :param np.ndarray weights: current weights
    :param float obj_func_eval: objective function value
    """
    # Print to the terminal.
    global global_current_step
    global_current_step += 1
    print(f"Iter {global_current_step}, current_value: {obj_func_eval}")

    # Save the metric.
    mlflow.log_metric(f"train_loss", obj_func_eval, step=global_current_step)

    # Save the fig.
    global global_objective_func_vals
    global_objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(global_objective_func_vals)), global_objective_func_vals)
    global GLOBAL_GRAPH_PATH
    plt.savefig(GLOBAL_GRAPH_PATH)
