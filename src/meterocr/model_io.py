"""Save and load trained model bundles using joblib."""

from pathlib import Path
from typing import Any

import joblib

from meterocr import __version__
from meterocr.types import HOGConfig, NormalizationConfig, TrainingConfig

_BUNDLE_VERSION = __version__


def save_model_bundle(
    model_path: Path,
    classifier: Any,
    hog_cfg: HOGConfig,
    normalization_cfg: NormalizationConfig,
    training_cfg: TrainingConfig,
    class_labels: list[int],
    scaler: Any | None = None,
) -> None:
    """Save a trained model and its configuration as a joblib bundle.

    Args:
        model_path: Destination path for the .joblib file.
        classifier: Fitted scikit-learn classifier.
        hog_cfg: HOG configuration used during training.
        normalization_cfg: Normalization configuration used during training.
        training_cfg: Training configuration.
        class_labels: Sorted list of class labels (digits 0-9).
        scaler: Fitted scaler or None.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "classifier": classifier,
        "scaler": scaler,
        "hog_cfg": hog_cfg,
        "normalization_cfg": normalization_cfg,
        "training_cfg": training_cfg,
        "class_labels": class_labels,
        "version": _BUNDLE_VERSION,
    }
    joblib.dump(bundle, model_path)


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load a model bundle from a joblib file.

    Args:
        model_path: Path to the .joblib file.

    Returns:
        Dictionary with keys: classifier, scaler, hog_cfg, normalization_cfg,
        training_cfg, class_labels, version.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    return joblib.load(model_path)
