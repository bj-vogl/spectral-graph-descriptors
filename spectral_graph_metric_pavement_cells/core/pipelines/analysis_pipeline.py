from typing import Dict
from spectral_graph_metric_pavement_cells.core.results.distance_result import DistanceResult
from spectral_graph_metric_pavement_cells.core.interfaces.analysis_strategy import AnalysisStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

class AnalysisPipeline:
    def __init__(self, analysis_strategy: AnalysisStrategy, analysis_feature_keys: Dict[str, str], plot: bool = True, save_path: str | None = None, config_name: str | None = None):
        self.analysis_strategy = analysis_strategy
        self.analysis_feature_keys = analysis_feature_keys
        self.results: Dict[str, DistanceResult] = {}
        self.plot = plot
        self.save_path = save_path
        self.config_name = config_name

        if hasattr(self.analysis_strategy, 'set_save_info'):
            self.analysis_strategy.set_save_info(save_path, config_name)

    def set_results(self, result_dict: Dict[str, DistanceResult]) -> None:
        self.results = result_dict

    def run_analysis(self):
        if not self.results:
            logger.info("No comparable results found.")
            return {}

        return self.analysis_strategy.analyze(self.results)
