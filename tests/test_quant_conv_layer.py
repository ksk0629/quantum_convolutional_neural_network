import pytest
import qiskit

from src.quant_conv_layer import QuantConvLayer


class TestQuantConvLayer:

    @classmethod
    def setup_class(cls):
        """Create and set the QuantConvLayer class as a member variable for later uses."""
        cls.num_qubits = 4
        cls.param_prefix = "theta"
        cls.quant_conv_layer = QuantConvLayer(
            num_qubits=cls.num_qubits, param_prefix=cls.param_prefix
        )
        assert cls.quant_conv_layer.num_qubits == cls.num_qubits
        assert cls.quant_conv_layer.param_prefix == cls.param_prefix
