from abc import ABC, abstractmethod


class BaseQuantLayer(ABC):
    """Base quantum layer class"""

    @abstractmethod
    def get_circuit(self):
        pass
