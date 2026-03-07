"""Tests for features.py."""

import numpy as np
import pytest

from meterocr.features import extract_feature_matrix, extract_hog_features
from meterocr.types import HOGConfig


def test_hog_feature_vector_stable_shape(hog_cfg, synthetic_digit_image):
    features = extract_hog_features(synthetic_digit_image, hog_cfg)
    # Run again and check shape is identical
    features2 = extract_hog_features(synthetic_digit_image, hog_cfg)
    assert features.shape == features2.shape


def test_hog_feature_vector_is_float32(hog_cfg, synthetic_digit_image):
    features = extract_hog_features(synthetic_digit_image, hog_cfg)
    assert features.dtype == np.float32


def test_hog_feature_vector_is_1d(hog_cfg, synthetic_digit_image):
    features = extract_hog_features(synthetic_digit_image, hog_cfg)
    assert features.ndim == 1


def test_extract_feature_matrix_shape(hog_cfg, synthetic_digit_image):
    n = 5
    images = [synthetic_digit_image] * n
    matrix = extract_feature_matrix(images, hog_cfg)
    assert matrix.ndim == 2
    assert matrix.shape[0] == n


def test_extract_feature_matrix_dtype(hog_cfg, synthetic_digit_image):
    images = [synthetic_digit_image, synthetic_digit_image]
    matrix = extract_feature_matrix(images, hog_cfg)
    assert matrix.dtype == np.float32
