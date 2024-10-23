from collections.abc import Callable

import numpy as np
import qiskit
import qiskit_algorithms
import qiskit_algorithms.optimizers
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils.loss_functions.loss_functions import Loss


class QNNTrainer:
    """QNN Trainer class"""

    def __init__(
        self,
        qnn: qiskit.primitives.BaseEstimator | qiskit.primitives.BaseSampler,
        optimiser: (
            None
            | qiskit_algorithms.optimizers.Optimizer
            | qiskit_algorithms.optimizers.Minimizer
        ),
        loss: str | Loss,
        train_data: np.typing.ArrayLike,
        train_labels: np.typing.ArrayLike,
        test_data: np.typing.ArrayLike,
        test_labels: np.typing.ArrayLike,
        initial_point: None | np.ndarray = None,
        callback: None | Callable[[np.ndarray, float], None] | None = None,
        seed: None | int = 91,
    ):
        """Iniitalise this class.

        :param qiskit.primitives.BaseEstimator | qiskit.primitives.BaseSampler qnn: QNN
        :param None  |  qiskit_algorithms.optimizers.Optimizer  |  qiskit_algorithms.optimizers.Minimizer optimiser: optimiser
        :param np.typing.ArrayLike train_data: train data
        :param np.typing.ArrayLike train_labels: train label
        :param np.typing.ArrayLike test_data: test data
        :param np.typing.ArrayLike test_labels: test label
        :param None | np.ndarray initial_point: initial point, defaults to None
        :param None | Callable[[np.ndarray, float], None] | None callback: callback, defaults to None
        :param None | int seed: random seed, defaults to 91
        """
        self.qnn = qnn
        self.optimiser = optimiser
        self.loss = loss
        self.initial_point = initial_point
        self.callback = callback

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.seed = seed
