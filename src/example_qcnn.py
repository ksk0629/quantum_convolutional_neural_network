import numpy as np
import qiskit
import qiskit_algorithms
from qiskit_machine_learning.neural_networks import EstimatorQNN

from src.quant_conv_layer import QuantConvLayer
from src.quant_pool_layer import QuantPoolLayer


class ExampleQCNN:
    """QCNN class used in the qiskit example:
    https://qiskit-community.github.io/qiskit-machine-learning/getting_started.html"""

    def __init__(self, data_size: int):
        """Initialise this QCNN.

        :param int data_size: data dimenstion
        """
        # Create the feature map.
        self.feature_map = qiskit.circuit.library.ZFeatureMap(data_size)

        # Create the ansatz.
        self.ansatz = qiskit.QuantumCircuit(data_size, name="Ansatz")
        self.ansatz.compose(
            QuantConvLayer(8, "conv1").get_circuit(),
            list(range(data_size)),
            inplace=True,
        )
        self.ansatz.compose(
            QuantPoolLayer([0, 1, 2, 3], [4, 5, 6, 7], "pool1").get_circuit(),
            list(range(data_size)),
            inplace=True,
        )
        self.ansatz.compose(
            QuantConvLayer(4, "conv2").get_circuit(),
            list(range(4, data_size)),
            inplace=True,
        )
        self.ansatz.compose(
            QuantPoolLayer([0, 1], [2, 3], "pool2").get_circuit(),
            list(range(4, data_size)),
            inplace=True,
        )
        self.ansatz.compose(
            QuantConvLayer(2, "conv3").get_circuit(),
            list(range(6, data_size)),
            inplace=True,
        )
        self.ansatz.compose(
            QuantPoolLayer([0], [1], "pool3").get_circuit(),
            list(range(6, data_size)),
            inplace=True,
        )

        # Combine the feature map and ansatz.
        self.circuit = qiskit.QuantumCircuit(data_size)
        self.circuit.compose(self.feature_map, range(data_size), inplace=True)
        self.circuit.compose(self.ansatz, range(data_size), inplace=True)

        self.observable = qiskit.quantum_info.SparsePauliOp.from_list(
            [("Z" + "I" * 7, 1)]
        )

        self.estimator = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
        )

        self.weights = qiskit_algorithms.utils.algorithm_globals.random.random(
            self.qnn.num_weights
        )

    def __call__(self, input_data: np.ndarray):
        """Return the result of the forward pass with the self.weights."""
        return self.estimator.forward(
            input_data,
            self.weights,
        )
