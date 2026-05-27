from typing import Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod


class FilterStrategy(ABC):
    @abstractmethod
    def filter(
            self,
            data: Dict[str, Any],
            meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Filters the input data (graphs or contours).

        Args:
            data: Either Dict[str, nx.Graph] OR Dict[str, Dict[str, Any]] (contours)
            meta: Optional metadata (e.g., cell_info). Required if filtering Graphs.

        Returns:
            (kept, dropped)
        """
        pass