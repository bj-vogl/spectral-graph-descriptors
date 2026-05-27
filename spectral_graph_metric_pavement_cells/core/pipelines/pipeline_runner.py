import os
from typing import Any, Optional, Union, List

import networkx as nx

from spectral_graph_metric_pavement_cells.core.features import weight_funcs
from spectral_graph_metric_pavement_cells.core.pipelines.computation_pipeline import ComputationPipeline
from spectral_graph_metric_pavement_cells.core.results.distance_result import DistanceResult
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.utils.contours import build_contours_from_images, build_contours_from_graphs, build_contours_from_lsm
from spectral_graph_metric_pavement_cells.visualization.cells import plot_contours_grid

logger = get_logger(__name__)


def run_pipeline(
    reader: Any,
    graph: Any,
    distance: Any,
    data_path: str,
    save_path: Optional[str],
    analysis_feature_key: str,
    manual_pixel_distance: Optional[float] = None,
    node_coverage: float = 0.65,
    plot_contour: bool = False,
    plot_distance: bool = False,
    weight_func: Optional[Union[str, List[str]]] = None,
    filter_strategy: Optional[Any] = None,
) -> DistanceResult:

    pipeline = ComputationPipeline(
        reader_strategy=reader,
        graph_strategy=graph,
        distance_strategy=distance,
        filter_strategy=filter_strategy,
    )

    plots_save_path = None
    if save_path:
        plots_save_path = os.path.join(save_path, "plots")

    weight_func_map = {
        "distance": weight_funcs.distance_weight,
        "distance_norm": weight_funcs.distance_norm_weight,
        "inv_distance_bounded": weight_funcs.inv_distance_bounded,
        "distance_rbf": weight_funcs.distance_rbf,
    }
    selected_weight_funcs = []
    if weight_func is not None:
        input_keys = [weight_func] if isinstance(weight_func, str) else weight_func

        for key in input_keys:
            if key in weight_func_map:
                selected_weight_funcs.append(weight_func_map[key])
            elif key.lower() == "none" or key is None:
                pass
            else:
                logger.warning(f"Warning: Weight function key '{key}' not found.")

    images = pipeline.run_reader(data_path)

    if isinstance(next(iter(images.values())), nx.Graph):
        # if input is a pickle graph
        collection = os.path.basename(os.path.normpath(data_path))
        for G in images.values():
            G.graph["collection_name"] = collection

        contours_dict = build_contours_from_graphs(images)

        if filter_strategy is not None:
            contours_dict, _ = filter_strategy.filter(contours_dict)
            images = {k: v for k, v in images.items() if k in contours_dict}

        if plot_contour:
            contours_dict_str_keys = {str(k): v for k, v in contours_dict.items()}
            plot_contours_grid(contours_dict_str_keys, save_path=plots_save_path)

        graphs, cell_info = pipeline.compute_graphs(images, save_path=plots_save_path)

        features = pipeline.compute_features(
            graphs,
            contours=None,
            weight_func=selected_weight_funcs,
            analysis_feature_key= analysis_feature_key
        )

        if distance is not None:
            logger.info(f"Computing distance matrix using {distance.__class__.__name__}...")
            distance_matrix = pipeline.compute_distance(
                features=features,
                analysis_feature_key=analysis_feature_key,
                save_path=save_path
            )
        else:
            distance_matrix = None

        contours_for_result = {str(k): v for k, v in contours_dict.items()}
        umx, umy = 1.0, 1.0
    else:
        # if input is a .roi-file or .jpeg-file
        if reader.__class__.__name__ == "RoiReader":
            # if reader is RoiReader, resolution will be extracted from the respective LSM-file
            contours_dict = build_contours_from_lsm(images)
        else:
            contours_dict = build_contours_from_images(images)

        if filter_strategy is not None:
            contours_dict, _ = filter_strategy.filter(contours_dict)

        if plot_contour:
            contours_dict_str_keys = {str(k): v for k, v in contours_dict.items()}
            plot_contours_grid(contours_dict_str_keys, save_path=plots_save_path)

        graphs, cell_info, contour_resampled = pipeline.compute_graphs(contours_dict, save_path=plots_save_path)

        collection = os.path.basename(os.path.normpath(data_path))
        for G in graphs.values():
            G.graph["collection_name"] = collection

        features = pipeline.compute_features(
            graphs,
            contours=contour_resampled,
            weight_func=selected_weight_funcs,
            analysis_feature_key= analysis_feature_key
        )

        if distance is not None:
            logger.info(f"Computing distance matrix using {distance.__class__.__name__}...")
            distance_matrix = pipeline.compute_distance(
                features=features,
                analysis_feature_key=analysis_feature_key,
                save_path=save_path
            )
        else:
            distance_matrix = None

        if images and len(contours_dict) > 0:
            first_fname = next(iter(contours_dict))
            umx, umy = contours_dict[first_fname]["resolution"]
        else:
            umx, umy = 1.0, 1.0

        contours_for_result = {str(k): {'contour': v['contour'], 'resolution': v['resolution']}
                               for k, v in contours_dict.items()}

    return DistanceResult(
        distance_matrix=distance_matrix,
        analysis_feature_key=analysis_feature_key,
        features=features,
        manual_pixel_distance=manual_pixel_distance,
        node_coverage=node_coverage,
        plot_contour=plot_contour,
        plot_distance=plot_distance,
        contours=contours_for_result,
        cell_info=cell_info,
        um_per_pixel=(umx, umy),
    )