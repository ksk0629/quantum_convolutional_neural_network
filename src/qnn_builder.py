import numpy as np
import qiskit
import qiskit_algorithms
from qiskit_machine_learning.neural_networks import EstimatorQNN

from src.quant_conv_layer import QuantConvLayer
from src.quant_pool_layer import QuantPoolLayer


class QNNBuilder:
    """QNN builder class"""

    def __init__(self, data_size: int):
        """Initialise this builder.

        :param int data_size: data size
        """
        self.data_size = data_size

    def get_example_qnn_estimator(self):
        # Create the feature map.
        feature_map = qiskit.circuit.library.ZFeatureMap(self.data_size)

        # Create the ansatz.
        ansatz = qiskit.QuantumCircuit(self.data_size, name="Ansatz")
        ansatz.compose(
            QuantConvLayer(8, "conv1").get_circuit(),
            list(range(self.data_size)),
            inplace=True,
        )
        ansatz.compose(
            QuantPoolLayer([0, 1, 2, 3], [4, 5, 6, 7], "pool1").get_circuit(),
            list(range(self.data_size)),
            inplace=True,
        )
        ansatz.compose(
            QuantConvLayer(4, "conv2").get_circuit(),
            list(range(4, self.data_size)),
            inplace=True,
        )
        ansatz.compose(
            QuantPoolLayer([0, 1], [2, 3], "pool2").get_circuit(),
            list(range(4, self.data_size)),
            inplace=True,
        )
        ansatz.compose(
            QuantConvLayer(2, "conv3").get_circuit(),
            list(range(6, self.data_size)),
            inplace=True,
        )
        ansatz.compose(
            QuantPoolLayer([0], [1], "pool3").get_circuit(),
            list(range(6, self.data_size)),
            inplace=True,
        )

        # Combine the feature map and ansatz.
        circuit = qiskit.QuantumCircuit(self.data_size)
        circuit.compose(feature_map, range(self.data_size), inplace=True)
        circuit.compose(ansatz, range(self.data_size), inplace=True)

        observable = qiskit.quantum_info.SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
