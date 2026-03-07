"""Load and validate meter and default configurations from YAML files."""

from pathlib import Path

import yaml

from meterocr.types import (
    DigitBox,
    HOGConfig,
    MeterConfig,
    NormalizationConfig,
    TrainingConfig,
)


def load_meter_configs(path: Path) -> dict[str, MeterConfig]:
    """Load per-meter configurations from a YAML file.

    Args:
        path: Path to the meters YAML file.

    Returns:
        Mapping from meter_id to MeterConfig.

    Raises:
        ValueError: If any meter config is invalid.
    """
    with path.open() as f:
        data = yaml.safe_load(f)

    configs: dict[str, MeterConfig] = {}
    for entry in data["meters"]:
        digit_boxes = [DigitBox(**b) for b in entry["digit_boxes"]]
        crop_source_box: DigitBox | None = None
        if "crop_source_box" in entry and entry["crop_source_box"]:
            crop_source_box = DigitBox(**entry["crop_source_box"])

        perspective_src: list[tuple[float, float]] | None = None
        if "perspective_src_points" in entry and entry["perspective_src_points"]:
            perspective_src = [tuple(p) for p in entry["perspective_src_points"]]

        alignment_ref: Path | None = None
        if "alignment_reference_path" in entry and entry["alignment_reference_path"]:
            alignment_ref = Path(entry["alignment_reference_path"])

        cfg = MeterConfig(
            meter_id=entry["meter_id"],
            aligned_width=entry["aligned_width"],
            aligned_height=entry["aligned_height"],
            digit_boxes=digit_boxes,
            threshold_mode=entry.get("threshold_mode", "otsu"),
            invert_binary=entry.get("invert_binary", True),
            crop_source_box=crop_source_box,
            perspective_src_points=perspective_src,
            alignment_reference_path=alignment_ref,
            max_translation_px=entry.get("max_translation_px", 0),
            inner_pad_x=entry.get("inner_pad_x", 0),
            inner_pad_y=entry.get("inner_pad_y", 0),
        )
        _validate_meter_config(cfg)
        configs[cfg.meter_id] = cfg

    return configs


def load_default_configs(path: Path) -> tuple[HOGConfig, NormalizationConfig, TrainingConfig]:
    """Load HOG, normalization, and training configurations from a YAML file.

    Args:
        path: Path to the defaults YAML file.

    Returns:
        Tuple of (HOGConfig, NormalizationConfig, TrainingConfig).
    """
    with path.open() as f:
        data = yaml.safe_load(f)

    hog_data = data.get("hog", {})
    hog_cfg = HOGConfig(
        image_width=hog_data.get("image_width", 40),
        image_height=hog_data.get("image_height", 64),
        orientations=hog_data.get("orientations", 9),
        pixels_per_cell=tuple(hog_data.get("pixels_per_cell", [4, 4])),
        cells_per_block=tuple(hog_data.get("cells_per_block", [2, 2])),
        block_norm=hog_data.get("block_norm", "L2-Hys"),
        transform_sqrt=hog_data.get("transform_sqrt", False),
    )

    norm_data = data.get("normalization", {})
    norm_cfg = NormalizationConfig(
        blur_kernel=norm_data.get("blur_kernel", 3),
        min_component_area=norm_data.get("min_component_area", 8),
        bbox_margin_px=norm_data.get("bbox_margin_px", 2),
        output_width=norm_data.get("output_width", 40),
        output_height=norm_data.get("output_height", 64),
        foreground=norm_data.get("foreground", "white"),
    )

    train_data = data.get("training", {})
    train_cfg = TrainingConfig(
        model_type=train_data.get("model_type", "linear_svc"),
        test_size=train_data.get("test_size", 0.2),
        group_by=train_data.get("group_by", "frame"),
        random_state=train_data.get("random_state", 42),
        knn_neighbors=train_data.get("knn_neighbors", 3),
        knn_weights=train_data.get("knn_weights", "distance"),
        svc_c=train_data.get("svc_c", 1.0),
        use_standard_scaler=train_data.get("use_standard_scaler", True),
    )

    return hog_cfg, norm_cfg, train_cfg


def get_meter_config(configs: dict[str, MeterConfig], meter_id: str) -> MeterConfig:
    """Look up a meter config by ID.

    Args:
        configs: Mapping returned by load_meter_configs.
        meter_id: The meter identifier.

    Returns:
        The corresponding MeterConfig.

    Raises:
        KeyError: If meter_id is not found.
    """
    if meter_id not in configs:
        raise KeyError(f"Unknown meter_id '{meter_id}'. Available: {list(configs.keys())}")
    return configs[meter_id]


def _validate_meter_config(cfg: MeterConfig) -> None:
    """Raise ValueError if the meter config is inconsistent."""
    for i, box in enumerate(cfg.digit_boxes):
        if box.x < 0 or box.y < 0:
            raise ValueError(f"Meter {cfg.meter_id} digit_box[{i}] has negative origin")
        if box.x + box.w > cfg.aligned_width:
            raise ValueError(
                f"Meter {cfg.meter_id} digit_box[{i}] extends beyond aligned_width"
            )
        if box.y + box.h > cfg.aligned_height:
            raise ValueError(
                f"Meter {cfg.meter_id} digit_box[{i}] extends beyond aligned_height"
            )

    if cfg.crop_source_box is not None and cfg.perspective_src_points is not None:
        raise ValueError(
            f"Meter {cfg.meter_id}: specify at most one of crop_source_box or perspective_src_points"
        )
