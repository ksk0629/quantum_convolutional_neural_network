from .base_quant_layer import BaseQuantLayer


class QuantConvLayer(BaseQuantLayer):
    """Quantum convolutional layer class"""

    def __init__(self):
        """Initialise this convolutional layer."""
        pass

    def __get_one_circuit(self):
        """Return the convolutional circuit."""
        pass

    def get_layer(self):
        """Return the convolutional layer as a qiskit.QuantumCircuit."""
        pass
