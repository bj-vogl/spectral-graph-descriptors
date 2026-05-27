from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from spectral_graph_metric_pavement_cells.core.interfaces.distances_strategy import DistanceStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)


class KSDistance(DistanceStrategy):
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
        if n < 2:
            logger.info("Only one cell loaded → no comparison possible")
            return np.zeros((n, n))

        # Compute distance matrix
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                stat, _ = ks_2samp(eig_sets[i], eig_sets[j])
                D[i, j] = stat
                D[j, i] = stat


        # Save CSV
        if save_path:
            out_dir = Path(save_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"ks_distance_{analysis_feature_key}.csv"

            df = pd.DataFrame(D, index=labels, columns=labels)
            df.to_csv(out_csv)
            logger.info(f"KS distance matrix saved to: {out_csv}")

        return D