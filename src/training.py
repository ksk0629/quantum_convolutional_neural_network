from collections.abc import Callable
import os

import mlflow
import numpy as np
import qiskit_algorithms
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils.loss_functions.loss_functions import Loss

from src.qnn_builder import QNNBuilder
from src.qnn_trainer import QNNTrainer
from src.utils import callback_print, callback_mlflow


def select_qnn(mode: str, data_size: int) -> EstimatorQNN | SamplerQNN:
    """Select QNN according to the given mode.

    :param str mode: strings to select one model
    :param int data_size: data size
    :raises ValueError: if the given mode does not match anything prepared
    :return EstimatorQNN | SamplerQNN: selected QNN
    """
    match mode:
        case "example_estimator":
            qnn = QNNBuilder().get_example_structure_estimator_qnn(data_size)
        case "exact_aer_estimator":
            qnn = QNNBuilder().get_exact_aer_estimator_qnn(data_size)
        case "noisy_aer_estimator":
            qnn = QNNBuilder().get_noisy_aer_estimator_qnn(data_size)
        case "example_sampler":
            qnn = QNNBuilder().get_example_structure_sampler_qnn(data_size)
        case "exact_aer_sampler":
            qnn = QNNBuilder().get_exact_aer_sampler_qnn(data_size)
        case "noisy_aer_sampler":
            qnn = QNNBuilder().get_noisy_aer_sampler_qnn(data_size)
        case _:
            msg = f"""Unknown mode, '{mode}', is given.
            Select one from the folloiwng ones.
            - example_estimator
            - exact_aer_estimator
            - noisy_aer_estimator
            - example_sampler
            - exact_aer_sampler
            - noisy_aer_sampler
            """
            raise ValueError(msg)

    return qnn


def select_optimiser(optimiser_str: str) -> qiskit_algorithms.optimizers.Optimizer:
    """Select optimiser according to the given optimiser_str.
    The optimiser selected is from qiskit_algorithms.optimizers.Optimizer.
    See  https://qiskit-community.github.io/qiskit-algorithms/apidocs/qiskit_algorithms.optimizers.html.

    :param str optimiser_str: optimiser string
    :return qiskit_algorithms.optimizers.Optimizer: optimiser
    """
    match optimiser_str:
        # --- Local optimisers ---
        case "adam":
            optimiser = qiskit_algorithms.optimizers.ADAM
        case "adgd":
            optimiser = qiskit_algorithms.optimizers.ADGS
        case "cg":
            optimiser = qiskit_algorithms.optimizers.CG
        case "cobyla":
            optimiser = qiskit_algorithms.optimizers.COBYLA
        case "l_bfgs_b":
            optimiser = qiskit_algorithms.optimizers.L_BFGS_B
        case "gsls":
            optimiser = qiskit_algorithms.optimizers.GSLS
        case "gradient_descent":
            optimiser = qiskit_algorithms.optimizers.GradientDescent
        case "gradient_descent_state":
            optimiser = qiskit_algorithms.optimizers.GradientDescentState
        case "nelder_mead":
            optimiser = qiskit_algorithms.optimizers.NELDER_MEAD
        case "nft":
            optimiser = qiskit_algorithms.optimizers.NFT
        case "p_bfgs":
            optimiser = qiskit_algorithms.optimizers.P_BFGS
        case "powell":
            optimiser = qiskit_algorithms.optimizers.POWELL
        case "slsqp":
            optimiser = qiskit_algorithms.optimizers.COBYLA
        case "spsa":
            optimiser = qiskit_algorithms.optimizers.SPSA
        case "qnspsa":
            optimiser = qiskit_algorithms.optimizers.QNSPSA
        case "tnc":
            optimiser = qiskit_algorithms.optimizers.TNC
        case "scipy_optimiser":
            optimiser = qiskit_algorithms.optimizers.SciPyOptimiser
        case "umda":
            optimiser = qiskit_algorithms.optimizers.UMDA
        case "bobyqa":
            optimiser = qiskit_algorithms.optimizers.BOBYQN
        case "imfil":
            optimiser = qiskit_algorithms.optimizers.IMFIL
        case "snobfit":
            optimiser = qiskit_algorithms.optimizers.SNOBFIT
        # --- global optimisers ---
        case "crs":
            optimiser = qiskit_algorithms.optimizers.CRS
        case "direct_l":
            optimiser = qiskit_algorithms.optimizers.DIRECT_L
        case "direct_l_rand":
            optimiser = qiskit_algorithms.optimizers.DIRECT_L_RAND
        case "esch":
            optimiser = qiskit_algorithms.optimizers.ESCH
        case "isres":
            optimiser = qiskit_algorithms.ISRES
        case _:
            optimiser = None
    return optimiser


def select_callback(
    callback_str: str,
) -> None | Callable[[np.ndarray, float], None] | None:
    """Select the callback function accodring to the given string.

    :param str callback_str: callback string
    :return None | Callable[[np.ndarray, float], None] | None: callback function
    """
    match callback_str:
        case "callback_print":
            callback = callback_print
        case "callback_mlflow":
            callback = callback_mlflow
    return callback


def train(
    experiment_name: str,
    run_name: str,
    train_data: np.typing.ArrayLike,
    train_labels: np.typing.ArrayLike,
    test_data: np.typing.ArrayLike,
    test_labels: np.typing.ArrayLike,
    mode: str,
    model_path: str,
    optimiser_str: str,
    loss: str | Loss,
    initial_point: None | np.ndarray = None,
    callback_str: None | str = None,
    optimiser_settings: None | dict = None,
    seed: None | int = 91,
):
    """Train the QNN and save the training process.

    :param str experiment_name: experiment name for mlflow
    :param str run_name: run name for mlflow
    :param np.typing.ArrayLike train_data: training data
    :param np.typing.ArrayLike train_labels: training labels
    :param np.typing.ArrayLike test_data: test data
    :param np.typing.ArrayLike test_labels: test labels
    :param str mode: QNN mode
    :param str model_path: path to model to save
    :param str optimiser_str: optimiser string
    :param str | Loss loss: loss
    :param None | np.ndarray initial_point: initial point, defaults to None
    :param None | str callback_str: callback string, defaults to None
    :param None | dict optimiser_settings: optimiser settings, defaults to None
    :param None | int seed: random seed, defaults to 91
    """
    # Get the QNN.
    qnn = select_qnn(mode=mode, data_size=len(train_data[0]))
    print(f"Built the QNN, given mode: {mode}.")

    # Get the callback.
    callback = select_callback(callback_str=callback_str)

    # Create the classifier.
    qnn_trainer = QNNTrainer(
        qnn=qnn,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        initial_point=initial_point,
        callback=callback,
        seed=seed,
    )
    print("Built the QNNTrainer.")

    # Create the directory to save the model.
    dir_path = os.path.dirname(model_path)
    if os.path.isdir(dir_path):
        os.makedirs(dir_path)
    # Fit the model.
    optimiser = select_optimiser(optimiser_str=optimiser_str)
    print(
        f"Get optimiser, given optimiser: {optimiser_str}, the instance: {optimiser}."
    )

    # Start mlflow.
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # Create datasets and save them into mlflow.
        train_dataset = mlflow.data.from_numpy(
            np.asarray(train_data), targets=np.asarray(train_labels)
        )
        mlflow.log_input(train_dataset, context="training")
        test_dataset = mlflow.data.from_numpy(
            np.asarray(test_data), targets=np.asarray(test_labels)
        )
        mlflow.log_input(test_dataset, context="test")
        # Save conifigs into mlflow.
        model_config = {
            "model_mode": mode,
        }
        mlflow.log_params(model_config)
        train_config = {
            "optimiser_str": optimiser_str,
            "loss": loss,
            "initial_point": initial_point,
            "optimiser_settings": optimiser_settings,
            "seed": seed,
        }
        mlflow.log_params(train_config)

        qnn_trainer.fit(
            model_path=model_path,
            optimiser=optimiser,
            loss=loss,
            optimiser_settings=optimiser_settings,
        )
