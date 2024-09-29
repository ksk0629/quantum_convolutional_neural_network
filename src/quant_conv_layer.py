from .base_quant_layer import BaseQuantLayer


class QuantConvLayer(BaseQuantLayer):
    """Quantum convolutional layer class"""

    def __init__(self, num_qubits: int, param_prefix: str):
        """Initialise this convolutional layer.

        :param int num_qubits: number of qubits
        :param str param_prefix: prefix strings of parameters
        :raises TypeError: if given num_qubits is not int type
        :raises ValueError: if given num_qubits is not non-zero and positive
        :raises TypeError: if given param_preifx is not str
        """
        if not isinstance(num_qubits, int):
            msg = f"num_qubits is must be int, but {type(num_qubits)}."
            raise TypeError(msg)
        if num_qubits < 1:
            msg = f"num_qubits is must be greater than one, but {num_qubits}."
            raise ValueError(msg)
        self.num_qubits = num_qubits

        if not isinstance(param_prefix, str):
            msg = f"param_prefix is must be str, but {type(param_prefix)}."
            raise TypeError(msg)
        self.param_prefix = param_prefix

    def __get_one_circuit(self):
        """Return the convolutional circuit."""
        pass

    def get_layer(self):
        """Return the convolutional layer as a qiskit.QuantumCircuit."""
        pass
