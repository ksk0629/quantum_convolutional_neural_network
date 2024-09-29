import numpy as np
import qiskit

from .base_quant_layer import BaseQuantLayer


class QuantConvLayer(BaseQuantLayer):
    """Quantum convolutional layer class"""

    def __init__(self, num_qubits: int, param_prefix: str):
        """Initialise this convolutional layer.

        :param int num_qubits: number of qubits
        :param str param_prefix: prefix strings of parameters
        :raises TypeError: if given num_qubits is not int type
        :raises ValueError: if given num_qubits is smaller than 2
        :raises TypeError: if given param_preifx is not str
        """
        if not isinstance(num_qubits, int):
            msg = f"num_qubits is must be int, but {type(num_qubits)}."
            raise TypeError(msg)
        if num_qubits < 2:
            msg = f"num_qubits is must be greater than one, but {num_qubits}."
            raise ValueError(msg)
        self.num_qubits = num_qubits

        if not isinstance(param_prefix, str):
            msg = f"param_prefix is must be str, but {type(param_prefix)}."
            raise TypeError(msg)
        self.param_prefix = param_prefix

    def __get_one_circuit(
        self, params: qiskit.circuit.ParameterVector
    ) -> qiskit.QuantumCircuit:
        """Return the convolutional circuit."""
        # Create the convolutional circuit.
        target = qiskit.QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)

        return target

    def get_circuit(self) -> qiskit.QuantumCircuit:
        """Return the convolutional layer as a qiskit.QuantumCircuit."""
        # Prepare variables to use later.
        qc = qiskit.QuantumCircuit(self.num_qubits, name="Convolutional Layer")
        qubits = list(range(self.num_qubits))
        num_params = self.num_qubits * 3 if self.num_qubits > 2 else 3
        params = qiskit.circuit.ParameterVector(self.param_prefix, length=num_params)

        # Add the convolutional unitary gates to each pair of two qubits.
        param_index = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(
                self.__get_one_circuit(params[param_index : (param_index + 3)]),
                [q1, q2],
            )
            qc.barrier()
            param_index += 3
        if self.num_qubits != 2:
            for q1, q2 in zip(
                qubits[1::2], qubits[2::2] + [0]
            ):  # [0]: To connect the last and first qubits.
                qc = qc.compose(
                    self.__get_one_circuit(params[param_index : (param_index + 3)]),
                    [q1, q2],
                )
                qc.barrier()
                param_index += 3

        qc_inst = qc.to_instruction()
        qc = qiskit.QuantumCircuit(self.num_qubits)
        qc.append(qc_inst, qubits)

        return qc
