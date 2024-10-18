import qiskit
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import BaseEstimatorV2

from src.quant_conv_layer import QuantConvLayer
from src.quant_pool_layer import QuantPoolLayer


class QNNBuilder:
    """QNN builder class"""

    def __init__(self):
        """Initialise this builder."""

    def get_example_estimator_qnn(self) -> EstimatorQNN:
        """Get the EstimatorQNN introduced in the qiskit example.

        :return EstimatorQNN: EstimatorQNN introduced in qiskit example
        """
        return self.get_example_structure_estimator_qnn(8)

    def get_example_exact_aer_estimator_qnn(self) -> EstimatorQNN:
        """Get the EstimatorWNN introduced in the qiskit example with the exact estimator from qiskit_aer.

        :return EstimatorQNN: EstimatorQNN introduced in qiskit example with the exact estimator from qiskit_aer.
        """
        return self.get_example_structure_estimator_qnn(8, AerEstimator())

    def get_example_structure_estimator_qnn(
        self, data_size: int, estimator: None | BaseEstimatorV2 = None
    ) -> EstimatorQNN:
        """Get the QCNN having the structure as follows.
        First, there is the ZFeatureMap,
        and then there are series of the ordered sets of the QuantConvLayer and QuantPoolLayer
        until the number of active qubits is one.

        :param int data_size: data size
        :param None | BaseEstimatorV2 estimator: estimator primitive, defaults to None
        :return EstimatorQNN: EstimatorQNN having structure introduced in qiskit example
        """
        feature_map = self.__get_z_feature_map(data_size=data_size)
        ansatz = self.__get_ansatz(data_size=data_size)

        # Combine the feature map and ansatz.
        circuit = qiskit.QuantumCircuit(data_size)
        circuit.compose(feature_map, range(data_size), inplace=True)
        circuit.compose(ansatz, range(data_size), inplace=True)

        observable = qiskit.quantum_info.SparsePauliOp.from_list(
            [("Z" + "I" * (data_size - 1), 1)]
        )

        return EstimatorQNN(
            estimator=estimator,
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )

    def __get_z_feature_map(self, data_size: int) -> qiskit.QuantumCircuit:
        """Get the quantum circuit representing ZFeatureMap.

        :param int data_size: data size
        :return qiskit.QuantumCircuit: quantum circuit representing ZFeatureMap
        """
        # Create the feature map.
        feature_map = qiskit.circuit.library.ZFeatureMap(data_size)
        feature_map.barrier()

        return feature_map

    def __get_ansatz(self, data_size: int) -> qiskit.QuantumCircuit:
        """Get the quantum circuit having the following structure;
        There are series of the ordered sets of the QuantConvLayer and QuantPoolLayer
        until the number of active qubits is one.

        :param int data_size: data size
        :return qiskit.QuantumCircuit: series of QuantConvLayer and QuantPoolLayer
        """
        # Create the ansatz.
        current_data_size = data_size
        ansatz = qiskit.QuantumCircuit(current_data_size, name="Ansatz")
        index = 1
        while current_data_size != 1:
            # Create the quantum convolutional layer.
            ansatz.compose(
                QuantConvLayer(current_data_size, f"conv{index}").get_circuit(),
                list(range(data_size - current_data_size, data_size)),
                inplace=True,
            )
            # Create the pooling layer.
            sources = range(current_data_size // 2)
            sinks = range(current_data_size // 2, current_data_size)
            ansatz.compose(
                QuantPoolLayer(sources, sinks, f"pool{index}").get_circuit(),
                list(range(data_size - current_data_size, data_size)),
                inplace=True,
            )
            # Reduce the data size.
            current_data_size //= 2
            # Update the index.
            index += 1

        return ansatz
