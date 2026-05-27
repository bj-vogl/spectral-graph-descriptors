import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

class MaxnormScaler(BaseEstimator, TransformerMixin):
    """
    Sample-wise Maxnorm Scaler: Scales each feature vector (row) individually
    by dividing every element in the vector by the maximum value
    found within that vector.

    This ensures the largest value in every sample is 1.0.
    Unlike global scalers, this operates on each sample independently.

    Parameters
    ----------
    eps : float
        Small value to avoid division by zero if a vector is all zeros.
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = float(eps)

    def fit(self, X, y=None):
        """
        No-op: This scaler calculates the max per sample during transform,
        so there are no global statistics to learn from the training set.
        """
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)

        # Determine the shape to handle both 1D (single sample) and 2D arrays
        if X_arr.ndim == 1:
            # Single vector case
            max_val = float(np.max(X_arr))
            if max_val <= self.eps:
                logger.warning(
                    f"Max value ({max_val}) is smaller than eps ({self.eps}). "
                    "Scaling factor set to 1.0."
                )
                scale = 1.0
            else:
                scale = max_val

            return X_arr / scale
        else:
            # Batch case: Calculate max value per row (axis=1)
            max_vals = np.max(X_arr, axis=1, keepdims=True)

            # Create a mask for values smaller than eps
            small_val_mask = max_vals < self.eps

            if np.any(small_val_mask):
                count = np.sum(small_val_mask)
                logger.warning(
                    f"Found {count} sample(s) with max value <= eps ({self.eps}). "
                    "Scaling factor set to 1.0 for these samples."
                )
                # Replace values < eps with 1.0 using the mask
                max_vals[small_val_mask] = 1.0

            return X_arr / max_vals

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

class GlobalIQRScaler(BaseEstimator, TransformerMixin):
    """
    Global robust scaler: compute a single center (median) and scale (IQR) across
    ALL elements of X (n_samples * n_features). Then apply (X - center) / scale
    elementwise. Less sensitive to outliers than mean/std global scaling.

    Parameters
    ----------
    quantile_range : tuple(float, float)
        Lower and upper percentile to use for the IQR (default (25, 75)).
    eps : float
        Small value to avoid zero-scale. If IQR <= eps we fall back to std or 1.0.
    """

    def __init__(self, quantile_range: tuple = (25.0, 75.0), eps: float = 1e-9):
        self.quantile_range = tuple(quantile_range)
        self.eps = float(eps)
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.size == 0:
            self.center_ = 0.0
            self.scale_ = 1.0
            return self

        flat = X_arr.ravel()
        self.center_ = float(np.median(flat))

        q_low, q_high = np.percentile(flat, [self.quantile_range[0], self.quantile_range[1]])
        iqr = float(q_high - q_low)

        if iqr > self.eps:
            self.scale_ = iqr
        else:
            # fallback to std if IQR too small
            std = float(np.std(flat))
            self.scale_ = std if std > self.eps else 1.0

        return self

    def transform(self, X):
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("GlobalIQRScaler is not fitted. Call fit() first.")
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.center_) / self.scale_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

class GlobalStandardScaler(BaseEstimator, TransformerMixin):
    """
    A scaler that computes a single global mean and std across ALL elements of X
    (i.e. across samples AND features) and applies (X - mean) / std elementwise.
    """

    def __init__(self, ddof: int = 0):
        # ddof aligns with numpy.std default behavior for population std (ddof=0).
        self.ddof = ddof
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        # Expect X to be 2D: (n_samples, n_features)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.size == 0:
            # avoid zero-division
            self.mean_ = 0.0
            self.scale_ = 1.0
            return self
        # compute global mean/std across all values
        self.mean_ = float(np.mean(X_arr.ravel()))
        # std with selected ddof; protect against zero std
        std = float(np.std(X_arr.ravel(), ddof=self.ddof))
        self.scale_ = std if std > 0.0 else 1.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("GlobalStandardScaler not fitted. Call fit() before transform().")
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)