from abc import ABC, abstractmethod
from typing import Any

class PersistenceStrategy(ABC):
    """
    Interface for saving and loading models.
    """
    @abstractmethod
    def save(self, model: Any, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> Any:
        pass