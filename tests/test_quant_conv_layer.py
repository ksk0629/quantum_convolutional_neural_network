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

    @pytest.mark.parametrize("num_qubits", [1, 2, 10])
    @pytest.mark.parametrize("param_prefix", ["", "param_prefix", "t e s t"])
    def test_init(self, num_qubits, param_prefix):
        """Normal test;
        Initialises the QuantConvLayer with normal arguments.

        Check if
        - the returned value has num_qubits variable begin the same as the argment.
        - the returned value has param_prefix begin the same as the argment.
        """
        quant_conv_layer = QuantConvLayer(
            num_qubits=num_qubits, param_prefix=param_prefix
        )
        assert quant_conv_layer.num_qubits == num_qubits
        assert quant_conv_layer.param_prefix == param_prefix

    @pytest.mark.parametrize("num_qubits", [1.1, [1], "1"])
    def test_init_with_non_int_num_qubits(self, num_qubits):
        """Abnormal test;
        Initialises the QuantConvLayer with non int num_qubits.

        Check if TypeError happens.
        """
        with pytest.raises(TypeError):
            QuantConvLayer(num_qubits=num_qubits, param_prefix=self.param_prefix)

    @pytest.mark.parametrize("num_qubits", [0, -1])
    def test_init_with_negative_num_qubits(self, num_qubits):
        """Abnormal test;
        Initialises the QuantConvLayer with non positive num_qubits.

        Check if ValueError happens.
        """
        with pytest.raises(ValueError):
            QuantConvLayer(num_qubits=num_qubits, param_prefix=self.param_prefix)

    @pytest.mark.parametrize("param_prefix", [1, 1.1, ["t e s t"]])
    def test_init_with_wrong_non_str_param_prefix(self, param_prefix):
        """Abnormal test;
        Initialises the QuantConvLayer with non str param_prefix.

        Check if TypeError happens.
        """
        with pytest.raises(TypeError):
            QuantConvLayer(num_qubits=self.num_qubits, param_prefix=param_prefix)

    @pytest.mark.parametrize("num_qubits", [1, 2, 10])
    def test_get_circuit(self, num_qubits):
        """Normal test;
        Run the get_circuit function.

        Check if
        - the return value is qiskit.QuantumCircuit.
        - the returned circuit has the parameters attribute.
        - the length of the returned circuit's parameters attribute is 3 * num_qubits
            if num_qubits is greater than 1. Otherwise 1.
        """
        quant_conv_layer = QuantConvLayer(
            num_qubits=num_qubits, param_prefix=self.param_prefix
        )
        conv_layer_circuit = quant_conv_layer.get_circuit()

        assert isinstance(conv_layer_circuit, qiskit.QuantumCircuit)
        params = conv_layer_circuit.parameters
        num_params = num_qubits * 3 if num_qubits > 1 else 1
        assert len(params) == num_params
