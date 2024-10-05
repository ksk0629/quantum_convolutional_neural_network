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

    @pytest.mark.parametrize("num_qubits", [2, 6, 10])
    @pytest.mark.parametrize("param_prefix", ["", "param_prefix", "t e s t"])
    def test_init(self, num_qubits, param_prefix):
        """Normal test;
        Initialises the QuantConvLayer with normal arguments.

        Check if
        - the returned value has num_qubits variable being the same as the argment.
        - the returned value has param_prefix being the same as the argment.
        """
        quant_conv_layer = QuantConvLayer(
            num_qubits=num_qubits, param_prefix=param_prefix
        )
        assert quant_conv_layer.num_qubits == num_qubits
        assert quant_conv_layer.param_prefix == param_prefix

    @pytest.mark.parametrize("num_qubits", [0, -1, 1])
    def test_init_with_smaller_than_2_num_qubits(self, num_qubits):
        """Abnormal test;
        Initialises the QuantConvLayer with non positive num_qubits.

        Check if ValueError happens.
        """
        with pytest.raises(ValueError):
            QuantConvLayer(num_qubits=num_qubits, param_prefix=self.param_prefix)

    @pytest.mark.parametrize("num_qubits", [3, 5, 7])
    def test_init_with_odd_num_qubits(self, num_qubits):
        """Abnormal test;
        Initialises the QuantConvLayer with odd num_qubits.

        Check if ValueError happens.
        """
        with pytest.raises(ValueError):
            QuantConvLayer(num_qubits=num_qubits, param_prefix=self.param_prefix)

    @pytest.mark.parametrize("num_qubits", [2, 6, 10])
    def test_get_circuit(self, num_qubits):
        """Normal test;
        Run the get_circuit function.

        Check if
        - the return value is qiskit.QuantumCircuit.
        - the returned circuit has the parameters attribute.
        - the length of the returned circuit's parameters attribute is 3 * num_qubits
            if num_qubits is greater than 2. Otherwise 1 * 3.
        """
        quant_conv_layer = QuantConvLayer(
            num_qubits=num_qubits, param_prefix=self.param_prefix
        )
        conv_layer_circuit = quant_conv_layer.get_circuit()

        assert isinstance(conv_layer_circuit, qiskit.QuantumCircuit)
        params = conv_layer_circuit.parameters
        num_params = num_qubits * 3 if num_qubits > 2 else 3
        assert len(params) == num_params
