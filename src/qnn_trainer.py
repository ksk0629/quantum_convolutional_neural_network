from collections.abc import Callable
import os

import numpy as np
import qiskit
import qiskit_algorithms
import qiskit_algorithms.optimizers
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils.loss_functions.loss_functions import Loss

import src.utils


class QNNTrainer:
    """QNN Trainer class"""

    def __init__(
        self,
        qnn: qiskit.primitives.BaseEstimator | qiskit.primitives.BaseSampler,
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
        :param np.typing.ArrayLike train_data: train data
        :param np.typing.ArrayLike train_labels: train label
        :param np.typing.ArrayLike test_data: test data
        :param np.typing.ArrayLike test_labels: test label
        :param None | np.ndarray initial_point: initial point, defaults to None
        :param None | Callable[[np.ndarray, float], None] | None callback: callback, defaults to None
        :param None | int seed: random seed, defaults to 91
        """
        self.qnn = qnn
        self.initial_point = initial_point
        self.callback = callback
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.seed = seed

    def fit(
        self,
        model_path: str,
        optimiser: (
            None
            | qiskit_algorithms.optimizers.Optimizer
            | qiskit_algorithms.optimizers.Minimizer
        ),
        loss: str | Loss,
        optimiser_settings: None | dict = None,
    ):
        """Fit the model with the settings.

        :param str model_path: path to fitted model to save
        :param None  |  qiskit_algorithms.optimizers.Optimizer  |  qiskit_algorithms.optimizers.Minimizer optimiser: optimiser
        :param str | Loss loss: loss
        :param None | dict optimiser_settings: optimiser settings, defaults to None
        """
        # Fix the random seeds according to the setting.
        if self.seed is not None:
            src.utils.fix_seed(self.seed)

        # Create an instance of NeuralNetworkClassifier.
        if optimiser is not None:
            _optimiser = optimiser(**optimiser_settings)
        else:
            _optimiser = None
        self.classifier = NeuralNetworkClassifier(
            neural_network=self.qnn,
            optimizer=_optimiser,
            loss=loss,
            initial_point=self.initial_point,
            callback=self.callback,
        )

        # Fit the data.
        train_x = np.asarray(self.train_data)
        train_y = np.asarray(self.train_labels)
        self.classifier.fit(train_x, train_y)

        # Get accuracy for the traininig data.
        train_accuracy = 100 * self.classifier.score(train_x, train_y)
        print(f"Accuracy for the training data: {train_accuracy}")

        # Get accuracy for the test data.
        test_x = np.asarray(self.test_data)
        test_y = np.asarray(self.test_labels)
        test_accuracy = 100 * self.classifier.score(test_x, test_y)
        print(f"Accuracy for the test data: {test_accuracy}")

        # Save the fitted model.
        dir_path = os.path.dirname(os.path.abspath(model_path))
        os.makedirs(dir_path, exist_ok=True)
        self.classifier.save(model_path)
