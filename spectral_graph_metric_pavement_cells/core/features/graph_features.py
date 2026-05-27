import sys
from typing import Any, Optional, Dict, Tuple, List, Callable

import networkx as nx
import numpy as np

from spectral_graph_metric_pavement_cells.core.features.weight_funcs import set_cell_rbf_sigma
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.utils.common import compute_sigma_median_pairwise_exact

logger = get_logger(__name__)


def weighted_adjacency_matrix(G: nx.Graph, weight_func=None) -> np.ndarray:
    """
    Compute a weighted adjacency matrix for a graph.
    Args:
        G: NetworkX Graph
        weight_func: function taking (pos_i, pos_j) -> weight
                     If None, uses the existing 'weight' attribute or 1.0
    Returns:
        np.ndarray: weighted adjacency matrix
    """
    n = G.number_of_nodes()
    A_weighted = np.zeros((n, n))
    pos = nx.get_node_attributes(G, 'pos')

    for i, j, data in G.edges(data=True):
        if weight_func is None:
            w = data.get('weight', 1.0)
        else:
            w = weight_func(pos[i], pos[j])
            #logger.info(f"weight: {w}")
        A_weighted[i, j] = w
        A_weighted[j, i] = w

    return A_weighted

def _build_unweighted_mats(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct A, D, L, L_sym for the unweighted case."""
    A = nx.to_numpy_array(G, weight=None)
    D = np.diag(A.sum(axis=1))
    L = D - A
    Q = D + A
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D))))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    Q_sym = D_inv_sqrt @ Q @ D_inv_sqrt
    return A, D, L, L_sym, Q, Q_sym

def _build_weighted_mats(G: nx.Graph, weight_func) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct A_w, D_w, L_w, L_w_sym for the weighted case."""
    A_w = weighted_adjacency_matrix(G, weight_func=weight_func)
    D_w = np.diag(A_w.sum(axis=1))
    L_w = D_w - A_w
    Q_w = D_w + A_w
    D_w_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D_w))))
    L_w_sym = D_w_inv_sqrt @ L_w @ D_w_inv_sqrt
    Q_w_sym = D_w_inv_sqrt @ Q_w @ D_w_inv_sqrt
    return A_w, D_w, L_w, L_w_sym, Q_w, Q_w_sym

def _eigh_sorted(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigen-decomposition and sort ascending by eigenvalues.
    """
    tolerance: float = 1e-14
    # square check
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Matrix must be square; got shape {M.shape}")

    # symmetry check
    diff = M - M.T
    max_asym = float(np.max(np.abs(diff)))
    if max_asym > tolerance:
        raise ValueError(
            "Matrix is not symmetric (required for np.linalg.eigh).\n"
            f"max_abs(M - M.T) = {max_asym:.3e} > 0\n"
        )
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]

def compute_spectral_features(graphs: Dict[Any, Any], resampled_contours, weight_funcs: Optional[List[Callable]] = None, analysis_feature_key: str = None) -> dict[Any, Any]:
    results: Dict[Any, Dict[str, Any]] = {}

    np.set_printoptions(threshold=sys.maxsize)

    for idx, G in graphs.items():
        # Unweighted
        A, D, L, L_sym, Q, Q_sym = _build_unweighted_mats(G)
        eigvals_L, eigvecs_L = _eigh_sorted(L)
        eigvals_Lsym, eigvecs_Lsym = _eigh_sorted(L_sym)
        eigvals_Lmaxnorm = np.sort(eigvals_L / (eigvals_L[-1]))
        eigvals_Q, eigvecs_Q = _eigh_sorted(Q)
        eigvals_Qsym, eigvecs_Qsym = _eigh_sorted(Q_sym)

        # Weighted
        if weight_funcs:
            c_vals_Lw = []
            c_vals_Lw_sym = []
            c_vals_Lw_maxnorm = []
            c_vals_Qw = []
            c_vals_Qw_sym = []
            A_w = D_w = L_w = L_w_sym = Q_w = Q_w_sym = None
            eigvecs_Lw = eigvecs_Lw_sym = eigvecs_Qw = eigvecs_Qw_sym = None

            for w_func in weight_funcs:
                # Handle RBF Sigma for this specific graph/contour
                if w_func.__name__ == "distance_rbf":
                    sigma = compute_sigma_median_pairwise_exact(resampled_contours[idx])
                    set_cell_rbf_sigma(sigma)

                _A_w, _D_w, _L_w, _L_w_sym, _Q_w, _Q_w_sym = _build_weighted_mats(G, w_func)

                _vals_Lw, _vecs_Lw = _eigh_sorted(_L_w)
                _vals_Lw_sym, _vecs_Lw_sym = _eigh_sorted(_L_w_sym)
                _vals_Lw_maxnorm = np.sort(_vals_Lw / (_vals_Lw[-1]))
                _vals_Qw, _vecs_Qw = _eigh_sorted(_Q_w)
                _vals_Qw_sym, _vecs_Qw_sym = _eigh_sorted(_Q_w_sym)

                c_vals_Lw.append(_vals_Lw)
                c_vals_Lw_sym.append(_vals_Lw_sym)
                c_vals_Lw_maxnorm.append(_vals_Lw_maxnorm)
                c_vals_Qw.append(_vals_Qw)
                c_vals_Qw_sym.append(_vals_Qw_sym)

                A_w, D_w, L_w, L_w_sym, Q_w, Q_w_sym= _A_w, _D_w, _L_w, _L_w_sym, _Q_w, _Q_w_sym
                eigvecs_Lw, eigvecs_Lw_sym, eigvecs_Qw, eigvecs_Qw_sym = _vecs_Lw, _vecs_Lw_sym, _vecs_Qw, _vecs_Qw_sym

            # If more than one weight function is given, concatenate eigenvalues
            eigvals_Lw = np.concatenate(c_vals_Lw)
            eigvals_Lw_sym = np.concatenate(c_vals_Lw_sym)
            eigvals_Lw_maxnorm = np.concatenate(c_vals_Lw_maxnorm)
            eigvals_Qw = np.concatenate(c_vals_Qw)
            eigvals_Qw_sym = np.concatenate(c_vals_Qw_sym)

        else:
            # No weight functions provided
            A_w = D_w = L_w = L_w_sym = Q_w = Q_w_sym = None
            eigvals_Lw = eigvecs_Lw = eigvals_Lw_sym = eigvecs_Lw_sym = eigvecs_Qw = eigvals_Qw = eigvecs_Qw_sym = eigvals_Qw_sym = None
            eigvals_Lw_maxnorm = None

        # Store Results
        results[idx] = {
            # Unweighted
            "A": A,
            "D": D,
            "L": L,
            "Q": Q,
            "L_sym": L_sym,
            "Q_sym": Q_sym,
            "eigvals_L": eigvals_L,
            "eigvecs_L": eigvecs_L,
            "eigvals_Lsym": eigvals_Lsym,
            "eigvecs_Lsym": eigvecs_Lsym,
            "eigvals_Lmaxnorm": eigvals_Lmaxnorm,
            "eigvals_Q": eigvals_Q,
            "eigvecs_Q": eigvecs_Q,
            "eigvals_Qsym": eigvals_Qsym,
            "eigvecs_Qsym": eigvecs_Qsym,

            # Weighted
            "A_w": A_w,
            "D_w": D_w,
            "L_w": L_w,
            "L_w_sym": L_w_sym,
            "Q_w": Q_w,
            "Q_w_sym": Q_w_sym,
            "eigvecs_Lw": eigvecs_Lw,
            "eigvecs_Lw_sym": eigvecs_Lw_sym,
            "eigvecs_Qw": eigvecs_Qw,
            "eigvecs_Qw_sym": eigvecs_Qw_sym,
            "eigvals_Lw": eigvals_Lw,
            "eigvals_Lw_sym": eigvals_Lw_sym,
            "eigvals_Lw_maxnorm": eigvals_Lw_maxnorm,
            "eigvals_Qw": eigvals_Qw,
            "eigvals_Qw_sym": eigvals_Qw_sym,
        }

        # Logging
        eigvals = None
        if analysis_feature_key is not None:
            eigvals = results[idx].get(analysis_feature_key)

        if eigvals is not None:
            pass

    return results
