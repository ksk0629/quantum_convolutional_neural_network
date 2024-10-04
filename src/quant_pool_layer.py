import numpy as np
import qiskit

from src.base_quant_layer import BaseQuantLayer


class QuantPoolLayer(BaseQuantLayer):
    """Quantum pooling layer class"""

    def __init__(self, sources: list[int], sinks: list[int], param_prefix: str):
        """Initialise this pooling layer.

        :param list[int] sources: list of positions of qubits will be remained
        :param list[int] sinks: list of positions of qubits will be discarded
        :param str param_prefix: prefix strings of parameters
        :raises ValueError: if the lengths of sources and sinks are different
        """
        if len(sources) != len(sinks):
            msg = f"sources and sinks must be the same length, but sources vs sinks = {len(sources)} vs {len(sinks)}."
            raise ValueError(msg)
        self.sources = sources
        self.sinks = sinks

        self.param_prefix = param_prefix

    def __get_pattern(
        self, params: qiskit.circuit.ParameterVector
    ) -> qiskit.QuantumCircuit:
        """Return the pooling circuit."""
        # Create the pooling circuit.
        target = qiskit.QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)

        return target

    def get_circuit(self) -> qiskit.QuantumCircuit:
        """Return the pooling layer as a qiskit.QuantumCircuit."""
        # Make parameters.
        num_qubits = len(self.sources) + len(self.sinks)
        param_index = 0
        params = qiskit.circuit.ParameterVector(
            self.param_prefix, length=num_qubits // 2 * 3
        )

        # Add each pooling circuit pattern to the circuit.
        qc = qiskit.QuantumCircuit(num_qubits, name="Pooling Layer")
        for source, sink in zip(self.sources, self.sinks):
            qc = qc.compose(
                self.__get_pattern(params[param_index : (param_index + 3)]),
                [source, sink],
            )
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()
        qc = qiskit.QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))

        return qc
