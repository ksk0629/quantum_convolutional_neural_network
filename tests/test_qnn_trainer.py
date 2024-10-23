import os

import numpy as np
import pytest
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN


from src.qnn_trainer import QNNTrainer


class TestQNNTrainer:
    @classmethod
    def setup_class(cls):
        """Create and set the QNNTrainer class as a member variable for later uses."""
        qc = QNNCircuit(num_qubits=2)

        cls.estimator_qnn = EstimatorQNN(circuit=qc)
        cls.sampler_qnn = SamplerQNN(circuit=qc)
        cls.optimiser = None
        cls.loss = "squared_error"
        cls.initial_point = None
        cls.callback = None

        train_data = np.asarray([[1, 2], [3, -4], [5, 6]])
        train_labels = np.asarray([1, -1, 1])
        test_data = np.asarray([[100, 1], [1, -2], [-10, 4]])
        test_labels = np.asarray([1, -1, -1])
        cls.train_data = train_data
        cls.train_labels = train_labels
        cls.test_data = test_data
        cls.test_labels = test_labels

        cls.seed = 91

        cls.qnn_trainer_estimator = QNNTrainer(
            qnn=cls.estimator_qnn,
            train_data=cls.train_data,
            train_labels=cls.train_labels,
            test_data=cls.test_data,
            test_labels=cls.test_labels,
            initial_point=cls.initial_point,
            callback=cls.callback,
            seed=cls.seed,
        )

        cls.qnn_trainer_sampler = QNNTrainer(
            qnn=cls.sampler_qnn,
            train_data=cls.train_data,
            train_labels=cls.train_labels,
            test_data=cls.test_data,
            test_labels=cls.test_labels,
            initial_point=cls.initial_point,
            callback=cls.callback,
            seed=cls.seed,
        )

    def test_setup(self):
        """Normal test;
        Check if the two trainer instances created in the setup_class method have correct member functions.
        """
        assert self.qnn_trainer_estimator.qnn == self.estimator_qnn
        assert np.allclose(self.qnn_trainer_estimator.train_data, self.train_data)
        assert np.allclose(self.qnn_trainer_estimator.train_labels, self.train_labels)
        assert np.allclose(self.qnn_trainer_estimator.test_data, self.test_data)
        assert self.qnn_trainer_estimator.initial_point == self.initial_point
        assert self.qnn_trainer_estimator.callback == self.callback
        assert self.qnn_trainer_estimator.seed == self.seed

        assert self.qnn_trainer_sampler.qnn == self.sampler_qnn
        assert np.allclose(self.qnn_trainer_sampler.train_data, self.train_data)
        assert np.allclose(self.qnn_trainer_sampler.train_labels, self.train_labels)
        assert np.allclose(self.qnn_trainer_sampler.test_data, self.test_data)
        assert self.qnn_trainer_sampler.initial_point == self.initial_point
        assert self.qnn_trainer_sampler.callback == self.callback
        assert self.qnn_trainer_sampler.seed == self.seed

    def test_fit(self, tmp_path):
        """Normal test;
        Run fit function of trainers created in the setup_class.

        Check if no error happens.
        """
        optimiser_settings = {"maxiter": 1}

        unique_estimator_path = f"{tmp_path}/estimator.model"
        self.qnn_trainer_estimator.fit(
            unique_estimator_path, self.optimiser, self.loss, optimiser_settings
        )
        NeuralNetworkClassifier.load(unique_estimator_path)

        unique_sampler_path = f"{tmp_path}/sampler.model"
        self.qnn_trainer_sampler.fit(
            unique_sampler_path, self.optimiser, self.loss, optimiser_settings
        )
        NeuralNetworkClassifier.load(unique_sampler_path)
