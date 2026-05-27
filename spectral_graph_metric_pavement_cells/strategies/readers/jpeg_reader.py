import os
from typing import Any
import cv2
import numpy as np
from spectral_graph_metric_pavement_cells.core.interfaces.readers_strategy import ReaderStrategy
from spectral_graph_metric_pavement_cells.core.runtime.logging_config import get_logger
logger = get_logger(__name__)

class JpegReader(ReaderStrategy):
    def __init__(self, default_resolution: float = 1.0):
        self.images = {}
        self.default_resolution = default_resolution  # µm/Pixel

    def read(self, path: str) -> dict[Any, Any]:
        """
        Returns: {filename: {'mask': binary_mask, 'resolution': (um_per_pixel_x, um_per_pixel_y)}}
        """
        self.images = {}
        found_jpeg = False

        # If the path is a file, handle it as a single-image input
        if os.path.isfile(path):
            fname = os.path.basename(path)
            if fname.lower().endswith((".jpg", ".jpeg")):
                found_jpeg = True
                file_path = path

                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"{fname} could not be loaded!")
                else:
                    # Create binary mask
                    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Invert if mask is mostly white
                    if np.count_nonzero(mask) > 0.98 * mask.size:
                        mask = cv2.bitwise_not(mask)

                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    # Set resolution to default
                    um_per_pixel_x = um_per_pixel_y = self.default_resolution

                    self.images[fname] = {
                        'mask': mask,
                        'resolution': (um_per_pixel_x, um_per_pixel_y)
                    }
            else:
                logger.warning(f"File {path} is not a JPEG.")
            return self.images

        for fname in os.listdir(path):
            if fname.lower().endswith((".jpg", ".jpeg")):
                found_jpeg = True
                file_path = os.path.join(path, fname)

                # Load image
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"{fname} could not be loaded!")
                    continue

                # Create binary mask
                _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Invert if mask is mostly white
                if np.count_nonzero(mask) > 0.98 * mask.size:
                    mask = cv2.bitwise_not(mask)

                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                # Set resolution to default
                um_per_pixel_x = um_per_pixel_y = self.default_resolution

                self.images[fname] = {
                    'mask': mask,
                    'resolution': (um_per_pixel_x, um_per_pixel_y)
                }

        if not found_jpeg:
            logger.warning(f"No JPEG found in {path}!")

        return self.images