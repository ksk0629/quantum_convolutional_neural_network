import yaml

import qiskit
import qiskit_aer
import qiskit_aer.primitives
import qiskit_ibm_runtime
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

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

    def get_exact_aer_estimator_qnn(
        self, data_size: int, seed: None | int = 91
    ) -> EstimatorQNN:
        """Get the EstimatorQNN with the exact estimator from qiskit_aer.

        :param int data_size: data size
        :param None | int seed: random seed
        :return EstimatorQNN: EstimatorQNN introduced in qiskit example with the exact estimator from qiskit_aer
        """
        return self.get_example_structure_estimator_qnn(
            data_size,
            qiskit_aer.primitives.Estimator(backend_options=dict(seed_simulator=seed)),
        )

    def get_noisy_aer_estimator_qnn(
        self, data_size: int, seed: None | int = 91
    ) -> EstimatorQNN:
        """Get the EstimatorQNN with a noisy estimator from qiskit_aer.

        :param int data_size: data size
        :param None | int seed: random seed
        :return EstimatorQNN: EstimatorQNN introduced in qiskit example with a noisy estimator from qiskit_aer
        """
        noise_model = qiskit_aer.noise.NoiseModel()
        cx_depolarizing_prob = 0.02
        noise_model.add_all_qubit_quantum_error(
            qiskit_aer.noise.depolarizing_error(cx_depolarizing_prob, 2), ["cx"]
        )

        noisy_estimator = qiskit_aer.primitives.Estimator(
            backend_options=dict(noise_model=noise_model, seed_simulator=seed)
        )

        return self.get_example_structure_estimator_qnn(data_size, noisy_estimator)

    def get_example_ibm_runtime_estimator_qnn(self, config_path: str) -> EstimatorQNN:
        """Get the EstimatorQNN introduced in the qiskit example with a real ibm quantum hardware.

        :param str config_path: path to config file including my ibm quantum token
        :return EstimatorQNN: EstimatorQNN introduced in qiskit example with real ibm quantum hardware
        """
        # Read my ibm quantum token.
        with open(config_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        ibm_quantum_token = config["my_ibm_quantum_token"]

        # Get the estimator being able to access to a real hardware.
        service = qiskit_ibm_runtime.QiskitRuntimeService(
            channel="ibm_quantum", token=ibm_quantum_token
        )
        backend = service.least_busy(operational=True, simulator=False)
        real_hardware_estimator = qiskit_ibm_runtime.Estimator(mode=backend)

        return self.get_example_structure_estimator_qnn(8, real_hardware_estimator)

    def get_example_structure_estimator_qnn(
        self, data_size: int, estimator: None | qiskit.primitives.BaseEstimator = None
    ) -> EstimatorQNN:
        """Get the QCNN having the structure as follows.
        First, there is the ZFeatureMap,
        and then there are series of the ordered sets of the QuantConvLayer and QuantPoolLayer
        until the number of active qubits is one.

        :param int data_size: data size
        :param None | BaseEstimator estimator: estimator primitive, defaults to None
        :return EstimatorQNN: EstimatorQNN having structure introduced in qiskit example
        """
        feature_map = self._get_z_feature_map(data_size=data_size)
        ansatz = self._get_ansatz(data_size=data_size)

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

    def get_example_sampler_qnn(self) -> SamplerQNN:
        """Get the SamplerQNN introduced in the qiskit example.

        :return SamplerQNN: EstimatorQNN introduced in qiskit example
        """
        return self.get_example_structure_sampler_qnn(8)

    def get_example_structure_sampler_qnn(
        self, data_size: int, sampler: None | qiskit.primitives.BaseSampler = None
    ) -> SamplerQNN:
        """Get the QCNN having the structure as follows.
        First, there is the ZFeatureMap,
        and then there are series of the ordered sets of the QuantConvLayer and QuantPoolLayer
        until the number of active qubits is one.

        :param int data_size: data size
        :param None | BaseSampler sampler: sampler primitive, defaults to None
        :return SamplerQNN: SamplerQNN having structure introduced in qiskit example
        """
        feature_map = self._get_z_feature_map(data_size=data_size)
        ansatz = self._get_ansatz(data_size=data_size)

        # Combine the feature map and ansatz.
        circuit = qiskit.QuantumCircuit(data_size)
        circuit.compose(feature_map, range(data_size), inplace=True)
        circuit.compose(ansatz, range(data_size), inplace=True)

        parity = lambda x: "{:b}".format(x).count("1") % 2
        output_shape = 2  # parity = 0, 1

        return SamplerQNN(
            sampler=sampler,
            circuit=circuit.decompose(),
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=output_shape,
        )

    def get_exact_aer_sampler_qnn(
        self, data_size: int, seed: None | int = 91
    ) -> SamplerQNN:
        """Get the SamplerQNN with the exact sampler from qiskit_aer.

        :param int data_size: data size
        :param None | int seed: random seed
        :return SamplerQNN: SamplerQNN introduced in qiskit example with the exact sampler from qiskit_aer
        """
        return self.get_example_structure_sampler_qnn(
            data_size,
            qiskit_aer.primitives.Sampler(backend_options=dict(seed_simulator=seed)),
        )

    def get_noisy_aer_sampler_qnn(
        self, data_size: int, seed: None | int = 91
    ) -> SamplerQNN:
        """Get the SamplerQNN with a noisy sampler from qiskit_aer.

        :param int data_size: data size
        :param None | int seed: random seed
        :return SamplerQNN: SamplerQNN introduced in qiskit example with a noisy sampler from qiskit_aer
        """
        noise_model = qiskit_aer.noise.NoiseModel()
        cx_depolarizing_prob = 0.02
        noise_model.add_all_qubit_quantum_error(
            qiskit_aer.noise.depolarizing_error(cx_depolarizing_prob, 2), ["cx"]
        )

        noisy_sampler = qiskit_aer.primitives.Sampler(
            backend_options=dict(noise_model=noise_model, seed_simulator=seed)
        )

        return self.get_example_structure_sampler_qnn(data_size, noisy_sampler)

    def _get_z_feature_map(self, data_size: int) -> qiskit.QuantumCircuit:
        """Get the quantum circuit representing ZFeatureMap.

        :param int data_size: data size
        :return qiskit.QuantumCircuit: quantum circuit representing ZFeatureMap
        """
        # Create the feature map.
        feature_map = qiskit.circuit.library.ZFeatureMap(data_size)
        feature_map.barrier()

        return feature_map

    def _get_ansatz(self, data_size: int) -> qiskit.QuantumCircuit:
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
