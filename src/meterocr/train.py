"""Model training: load dataset, extract features, fit classifier, save bundle."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from meterocr import dataset
from meterocr.features import extract_feature_matrix
from meterocr.model_io import save_model_bundle
from meterocr.types import HOGConfig, NormalizationConfig, TrainingConfig


def build_training_dataframe(samples_csv: Path) -> pd.DataFrame:
    """Load and filter samples suitable for training.

    Only 'ok' quality samples with valid labels are included.

    Args:
        samples_csv: Path to samples.csv.

    Returns:
        Filtered DataFrame.
    """
    df = dataset.load_samples(samples_csv)
    df = df[df["quality"] == "ok"].copy()
    df = df[df["digit_label"].between(0, 9)].copy()
    return df.reset_index(drop=True)


def split_train_validation(
    df: pd.DataFrame,
    cfg: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and validation sets without frame leakage.

    Groups are determined by cfg.group_by ('frame', 'date', or 'meter'),
    then the last fraction (cfg.test_size) of sorted groups form validation.

    Args:
        df: Training DataFrame.
        cfg: Training configuration.

    Returns:
        (train_df, val_df) tuple.
    """
    if cfg.group_by == "frame":
        groups = sorted(df["frame_id"].unique())
    elif cfg.group_by == "date":
        groups = sorted(df["timestamp"].dt.date.astype(str).unique())
        df = df.copy()
        df["_group"] = df["timestamp"].dt.date.astype(str)
    elif cfg.group_by == "meter":
        groups = sorted(df["meter_id"].unique())
    else:
        groups = sorted(df["frame_id"].unique())

    n_val = max(1, int(len(groups) * cfg.test_size))
    val_groups = set(groups[-n_val:])
    train_groups = set(groups[:-n_val])

    group_col = {
        "frame": "frame_id",
        "date": "_group",
        "meter": "meter_id",
    }.get(cfg.group_by, "frame_id")

    if group_col == "_group" and "_group" not in df.columns:
        df = df.copy()
        df["_group"] = df["timestamp"].dt.date.astype(str)

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    return train_df, val_df


def _load_images(df: pd.DataFrame, hog_cfg: HOGConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized images and labels from a DataFrame.

    Skips rows whose image file is missing or unreadable.

    Args:
        df: DataFrame with 'normalized_path' and 'digit_label' columns.
        hog_cfg: HOG config for output size reference.

    Returns:
        (feature_matrix, labels) as numpy arrays.
    """
    images = []
    labels = []
    for _, row in df.iterrows():
        img = cv2.imread(str(row["normalized_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (hog_cfg.image_width, hog_cfg.image_height))
        images.append(img)
        labels.append(int(row["digit_label"]))

    if not images:
        raise ValueError("No valid images found in training DataFrame")

    X = extract_feature_matrix(images, hog_cfg)
    y = np.array(labels, dtype=np.int32)
    return X, y


def train_knn(
    train_df: pd.DataFrame,
    hog_cfg: HOGConfig,
    normalization_cfg: NormalizationConfig,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    """Train a k-Nearest Neighbours classifier.

    Args:
        train_df: Training samples DataFrame.
        hog_cfg: HOG configuration.
        normalization_cfg: Normalization configuration (unused at this stage).
        cfg: Training configuration.

    Returns:
        Dict with keys: classifier, scaler, train_count, feature_dim, class_labels.
    """
    X, y = _load_images(train_df, hog_cfg)
    scaler = None
    if cfg.use_standard_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    clf = KNeighborsClassifier(n_neighbors=cfg.knn_neighbors, weights=cfg.knn_weights)
    clf.fit(X, y)
    return {
        "classifier": clf,
        "scaler": scaler,
        "train_count": len(y),
        "feature_dim": X.shape[1],
        "class_labels": sorted(int(c) for c in np.unique(y)),
    }


def train_linear_svc(
    train_df: pd.DataFrame,
    hog_cfg: HOGConfig,
    normalization_cfg: NormalizationConfig,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    """Train a LinearSVC classifier.

    Args:
        train_df: Training samples DataFrame.
        hog_cfg: HOG configuration.
        normalization_cfg: Normalization configuration (unused at this stage).
        cfg: Training configuration.

    Returns:
        Dict with keys: classifier, scaler, train_count, feature_dim, class_labels.
    """
    X, y = _load_images(train_df, hog_cfg)
    scaler = None
    if cfg.use_standard_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    clf = LinearSVC(C=cfg.svc_c, random_state=cfg.random_state, max_iter=2000)
    clf.fit(X, y)
    return {
        "classifier": clf,
        "scaler": scaler,
        "train_count": len(y),
        "feature_dim": X.shape[1],
        "class_labels": sorted(int(c) for c in np.unique(y)),
    }


def train_and_save_model(
    samples_csv: Path,
    model_path: Path,
    hog_cfg: HOGConfig,
    normalization_cfg: NormalizationConfig,
    training_cfg: TrainingConfig,
) -> dict[str, Any]:
    """Train a model on all 'ok' samples and save the bundle.

    Args:
        samples_csv: Path to samples.csv.
        model_path: Destination path for the model bundle.
        hog_cfg: HOG configuration.
        normalization_cfg: Normalization configuration.
        training_cfg: Training configuration.

    Returns:
        Dict with training results (classifier, scaler, train_count, feature_dim,
        class_labels).
    """
    df = build_training_dataframe(samples_csv)
    train_df, _ = split_train_validation(df, training_cfg)

    if training_cfg.model_type == "knn":
        result = train_knn(train_df, hog_cfg, normalization_cfg, training_cfg)
    else:
        result = train_linear_svc(train_df, hog_cfg, normalization_cfg, training_cfg)

    save_model_bundle(
        model_path=model_path,
        classifier=result["classifier"],
        hog_cfg=hog_cfg,
        normalization_cfg=normalization_cfg,
        training_cfg=training_cfg,
        class_labels=result["class_labels"],
        scaler=result["scaler"],
    )
    return result
