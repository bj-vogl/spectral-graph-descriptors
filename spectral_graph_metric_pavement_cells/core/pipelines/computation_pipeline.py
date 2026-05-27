# core/computation_pipeline.py
import os
from typing import Any

from spectral_graph_metric_pavement_cells.core.features.graph_features import compute_spectral_features
from spectral_graph_metric_pavement_cells.core.runtime.context import ReaderContext, DistanceContext, GraphContext, \
    FilterContext


class ComputationPipeline:
    def __init__(self, reader_strategy: Any, distance_strategy: Any, graph_strategy: Any, filter_strategy: Any = None):
        self.reader_ctx = ReaderContext(reader_strategy)
        self.distance_ctx = DistanceContext(distance_strategy)
        self.graph_ctx = GraphContext(graph_strategy)
        self.filter_ctx = FilterContext(filter_strategy) if filter_strategy is not None else None

        self.data = None
        self.graphs = {}
        self.features = {}
        self.result = None

    def run_reader(self, *args, **kwargs):
        """Execute reader and attach a collection_name derived from data_path."""
        self.data = self.reader_ctx.execute(*args, **kwargs)

        data_path = kwargs.get("data_path")
        if data_path is None and args:
            if isinstance(args[0], str):
                data_path = args[0]

        if isinstance(self.data, dict) and isinstance(data_path, str) and data_path:
            collection = os.path.basename(os.path.normpath(data_path))
            for _k, meta in self.data.items():
                if isinstance(meta, dict):
                    meta.setdefault("collection_name", collection)

        return self.data

    def compute_graphs(self, data, save_path=None):
        self.graphs = self.graph_ctx.execute(data, save_path=save_path)
        return self.graphs

    def compute_features(self, graphs, contours, weight_func= None, analysis_feature_key= None):
        self.features = compute_spectral_features(graphs, contours, weight_funcs=weight_func, analysis_feature_key=analysis_feature_key)
        return self.features

    def apply_filter(self, graphs, cell_info):
        if self.filter_ctx is None:
            return graphs, {}
        return self.filter_ctx.execute(graphs, cell_info)

    def compute_distance(self, features, **kwargs):
        if self.features is None:
            raise ValueError("Features have to be computed first!")
        self.result = self.distance_ctx.execute(features, **kwargs)
        return self.result