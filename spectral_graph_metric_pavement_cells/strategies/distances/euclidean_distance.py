from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from spectral_graph_metric_pavement_cells.core.interfaces.distances_strategy import DistanceStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)


class EuclideanDistance(DistanceStrategy):
    """ This code is not actively used"""

    def compute_distance(self, features: dict[str, Any], analysis_feature_key: str, save_path: str | None = None) -> np.ndarray:
        eig_sets = []
        labels = []
        for idx, res in features.items():
            if analysis_feature_key not in res:
                raise KeyError(f"Feature {idx} does not contain '{analysis_feature_key}'")
            eig_sets.append(np.sort(res[analysis_feature_key]))
            labels.append(idx)

        n = len(eig_sets)
        if n < 1:
            logger.info("No cell loaded → no comparison possible")
            return np.zeros((n, n))
        elif n < 2:
            logger.info("Only one cell loaded → no comparison possible")
            return np.zeros((n, n))

        # Ensure all spectra have the same length (pad with zeros if necessary)
        max_len = max(len(arr) for arr in eig_sets)
        eig_sets = [np.pad(arr, (0, max_len - len(arr))) for arr in eig_sets]

        # Distance matrix
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(eig_sets[i] - eig_sets[j])
                D[i, j] = dist
                D[j, i] = dist

        # Save CSV
        if save_path:
            out_dir = Path(save_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"euclidean_distance_{analysis_feature_key}.csv"

            df = pd.DataFrame(D, index=labels, columns=labels)
            df.to_csv(out_csv)
            logger.info(f"Euclidean distance matrix saved to: {out_csv}")

        return D