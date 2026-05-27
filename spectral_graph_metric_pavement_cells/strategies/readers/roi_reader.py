import os

import numpy as np
from typing import Any, Optional, List, Dict
from read_roi import read_roi_file
import cv2
from spectral_graph_metric_pavement_cells.core.interfaces.readers_strategy import ReaderStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
from spectral_graph_metric_pavement_cells.utils.lsm_meta import attach_lsm_path_to_images, extract_pixel_size_from_lsm

logger = get_logger(__name__)

class RoiReader(ReaderStrategy):
    """
    Reader for ImageJ/Fiji ROI-files.
    """

    def __init__(self, mask_shape=(1024, 1024), include_files: Optional[List[str]] = None, name_map: Optional[Dict[str, str]] = None):
        self.images = {}
        self.mask_shape = mask_shape
        self.include_files = include_files
        self.name_map = name_map or {}

    def _contour_to_mask(self, contour: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.mask_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
        return mask

    def _read_one_roi_file(self, file_path: str) -> None:
        """Read a single .roi file and append entries into self.images."""
        fname = os.path.basename(file_path)
        abs_path = os.path.abspath(file_path)
        try:
            rois = read_roi_file(file_path)
        except Exception as e:
            logger.warning(f"{fname} could not be read ({e})")
            return

        # Use display name from name_map if available; otherwise fallback to basename.
        display_key = self.name_map.get(abs_path, fname)

        for _, roi_data in rois.items():
            if roi_data.get('type') in ('polygon', 'freehand'):
                self.images[display_key] = {}
                contour = np.column_stack((roi_data['x'], roi_data['y']))
                mask = self._contour_to_mask(contour)
                lsm_path= attach_lsm_path_to_images(self.images, file_path, display_key)
                (x_px, y_px) = extract_pixel_size_from_lsm(lsm_path)
                x_um = x_px * 1e6 # convert to micrometers
                y_um = y_px * 1e6 # convert to micrometers
                self.images[display_key]['mask'] = mask
                self.images[display_key]['resolution'] = (x_um, y_um)
                self.images[display_key]['raw_contour'] = contour

    def read(self, path: str) -> dict[str, Any]:
        self.images = {}
        found_roi = False

        # 1) Explicit file mode
        if self.include_files:
            for p in self.include_files:
                if str(p).lower().endswith(".roi") and os.path.isfile(p):
                    found_roi = True
                    self._read_one_roi_file(p)
                    #logger.info(f"Read ROI-file: {p}")
            if not found_roi:
                logger.warning("No ROI-files from 'include_files' could be read!")
            return self.images

        # 2) If 'path' is a single .roi file, read it directly
        if path.lower().endswith(".roi") and os.path.isfile(path):
            found_roi = True
            self._read_one_roi_file(path)
            return self.images

        if not os.path.isdir(path):
            logger.warning(f"Path is not a folder: {path}")
            return self.images

        # 3) If 'path' is a directory, read all .roi files in it
        for fname in os.listdir(path):
            if fname.lower().endswith(".roi"):
                found_roi = True
                file_path = os.path.join(path, fname)
                self._read_one_roi_file(file_path)

        if not found_roi:
            logger.warning(f"No ROI-files found in folder {path}!")

        return self.images