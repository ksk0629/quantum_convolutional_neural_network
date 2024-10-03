import pytest

from src.example_qcnn import ExampleQCNN


class TestExampleQCNN:

    @classmethod
    def setup_class(cls):
        """Create and set the ExampleQCNN class as a member variable for later uses."""
        cls.example_qcnn = ExampleQCNN()
