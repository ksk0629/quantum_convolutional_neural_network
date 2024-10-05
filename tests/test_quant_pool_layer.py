import pytest
import qiskit

from src.quant_pool_layer import QuantPoolLayer


class TestQuantPoolLayer:

    @classmethod
    def setup_class(cls):
        """Create and set the QuantPoolLayer class as a member variable for later uses."""
        length = 4
        cls.sources = list(range(length))
        cls.sinks = list(range(length, length * 2))
        cls.param_prefix = "theta"
        cls.quant_pool_layer = QuantPoolLayer(
            sources=cls.sources, sinks=cls.sinks, param_prefix=cls.param_prefix
        )
        assert cls.quant_pool_layer.sources == cls.sources
        assert cls.quant_pool_layer.sinks == cls.sinks
        assert cls.quant_pool_layer.param_prefix == cls.param_prefix

    @pytest.mark.parametrize(
        "sources_and_sinks", [([0], [1]), ([0, 1], [2, 3]), ([0, 1, 2], [3, 4, 5])]
    )
    @pytest.mark.parametrize("param_prefix", ["", "param_prefix", "t e s t"])
    def test_init(self, sources_and_sinks, param_prefix):
        """Normal test;
        Initialises the QuantPoolLayer with normal arguments.

        Check if
        - the returned value has sources variable being the same as the argment.
        - the returned value has sinks variable being the same as the argment.
        - the returned value has param_prefix variable being the same as the argment.
        """
        (sources, sinks) = sources_and_sinks
        quant_pool_layer = QuantPoolLayer(
            sources=sources, sinks=sinks, param_prefix=param_prefix
        )
        assert quant_pool_layer.sources == sources
        assert quant_pool_layer.sinks == sinks
        assert quant_pool_layer.param_prefix == param_prefix

    @pytest.mark.parametrize("sources_and_sinks", [([0], [1, 2]), ([0, 1], [2])])
    def test_init_with_diff_lengths_sources_and_sinks(self, sources_and_sinks):
        """Abnormal test;
        Initialises the QuantPoolLayer with sources and sinks having different lengths.

        Check if ValueError happens.
        """
        (sources, sinks) = sources_and_sinks
        with pytest.raises(ValueError):
            QuantPoolLayer(sources=sources, sinks=sinks, param_prefix=self.param_prefix)

    @pytest.mark.parametrize(
        "sources_and_sinks", [([0], [1]), ([0, 1], [2, 3]), ([0, 1, 2], [3, 4, 5])]
    )
    def test_get_circuit(self, sources_and_sinks):
        """Normal test;
        Run the get_circuit function.

        Check if
        - the return value is qiskit.QuantumCircuit.
        - the returned circuit has the parameters attribute.
        - the length of the returned circuit's parameters attribute is (sources + sinks) // 2 * 3
        """
        (sources, sinks) = sources_and_sinks
        quant_pool_layer = QuantPoolLayer(
            sources=sources, sinks=sinks, param_prefix=self.param_prefix
        )
        pool_layer_circuit = quant_pool_layer.get_circuit()

        assert isinstance(pool_layer_circuit, qiskit.QuantumCircuit)
        params = pool_layer_circuit.parameters
        num_qubits = len(sources) + len(sinks)
        num_params = num_qubits // 2 * 3
        assert len(params) == num_params
