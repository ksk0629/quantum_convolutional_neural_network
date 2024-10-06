import qiskit
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

    def get_example_estimator_qnn(self) -> EstimatorQNN:
        """Get the EstimatorQNN introduced in the qiskit example.

        :return EstimatorQNN: EstimatorQNN introduced in qiskit example
        """
        return self.get_example_structure_etimator_qcnn(8)

    def get_example_structure_etimator_qcnn(self) -> EstimatorQNN:
        """Get the QCNN having the structure as follows.
        First, there is the ZFeatureMap,
        and then there are series of the ordered sets of the QuantConvLayer and QuantPoolLayer
        until the number of active qubits is one.
        """
        # Create the feature map.
        feature_map = qiskit.circuit.library.ZFeatureMap(self.data_size)
        feature_map.barrier()

        # Create the ansatz.
        current_data_size = self.data_size
        ansatz = qiskit.QuantumCircuit(current_data_size, name="Ansatz")
        index = 1
        while current_data_size != 1:
            # Create the quantum convolutional layer.
            ansatz.compose(
                QuantConvLayer(current_data_size, f"conv{index}").get_circuit(),
                list(range(self.data_size - current_data_size, self.data_size)),
                inplace=True,
            )
            # Create the pooling layer.
            sources = range(current_data_size // 2)
            sinks = range(current_data_size // 2, current_data_size)
            ansatz.compose(
                QuantPoolLayer(sources, sinks, f"pool{index}").get_circuit(),
                list(range(self.data_size - current_data_size, self.data_size)),
                inplace=True,
            )
            # Reduce the data size.
            current_data_size //= 2
            # Update the index.
            index += 1

        # Combine the feature map and ansatz.
        circuit = qiskit.QuantumCircuit(self.data_size)
        circuit.compose(feature_map, range(self.data_size), inplace=True)
        circuit.compose(ansatz, range(self.data_size), inplace=True)

        observable = qiskit.quantum_info.SparsePauliOp.from_list(
            [("Z" + "I" * (self.data_size - 1), 1)]
        )

        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
