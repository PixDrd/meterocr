"""HOG feature extraction for normalized digit images."""

import numpy as np
from skimage.feature import hog

from meterocr.types import FeatureVectorF32, HOGConfig, ImageU8


def extract_hog_features(normalized_img: ImageU8, cfg: HOGConfig) -> FeatureVectorF32:
    """Extract a HOG feature vector from a normalized digit image.

    Args:
        normalized_img: Grayscale normalized digit image of size
            (cfg.image_height, cfg.image_width).
        cfg: HOG configuration.

    Returns:
        Float32 HOG feature vector.
    """
    features = hog(
        normalized_img,
        orientations=cfg.orientations,
        pixels_per_cell=cfg.pixels_per_cell,
        cells_per_block=cfg.cells_per_block,
        block_norm=cfg.block_norm,
        transform_sqrt=cfg.transform_sqrt,
        feature_vector=True,
    )
    return features.astype(np.float32)


def extract_feature_matrix(images: list[ImageU8], cfg: HOGConfig) -> np.ndarray:
    """Extract HOG features from a list of images into a 2D matrix.

    Args:
        images: List of normalized grayscale digit images.
        cfg: HOG configuration.

    Returns:
        Array of shape (n_samples, n_features) in float32.
    """
    rows = [extract_hog_features(img, cfg) for img in images]
    return np.stack(rows, axis=0).astype(np.float32)
