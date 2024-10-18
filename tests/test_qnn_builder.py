import pytest

from src.qnn_builder import QNNBuilder


class TestQNNBuilder:

    @classmethod
    def setup_class(cls):
        """Create and set the QNNuilder class as a member variable for later uses."""
        cls.qnn_builder = QNNBuilder()

    def test_get_example_estimator_qnn(self):
        """Normal test;
        Runs get_example_estimator_qnn.

        Check if no error happens.
        """
        self.qnn_builder.get_example_estimator_qnn()

    @pytest.mark.parametrize("data_size", [2, 4, 16])
    def test_get_example_structure_estimator_qnn(self, data_size):
        """Normal test;
        Runs get_example_structure_estimator_qnn with valid data_size.

        Check if no error happens.
        """
        self.qnn_builder.get_example_structure_estimator_qnn(data_size)

    def test_get_example_aer_exact_estimator_qnn(self):
        """Normal test;
        Runs get_example_aer_exact_estimator_qnn.

        Check if no error happens.
        """
        self.qnn_builder.get_example_aer_exact_estimator_qnn()
