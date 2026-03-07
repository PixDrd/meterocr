# Water Meter OCR Implementation Spec
Version: v1
Target: Codex / Claude implementation handoff
Language: Python 3.11+
Primary libraries:
- opencv-python
- numpy
- scikit-image
- scikit-learn
- joblib
- pydantic or dataclasses only if useful
- typer or argparse for CLI

This document is a concrete implementation spec for the water meter OCR plan.
It adds:
- a recommended repository/file layout
- exact module responsibilities
- exact function signatures
- data formats
- model bundle contents
- a practical implementation order

The goal is to make the first implementation straightforward and low-risk.

---

## 1. Scope

Implement a Python package that reads 3 indoor water meters from images or a video stream.

System characteristics:
- each meter is photographed from a mostly fixed camera
- lighting is fairly stable
- the digit wheels have fixed positions
- the meter value is monotonic non-decreasing
- no deep learning / CNN training
- no full-line OCR engine
- shared pooled digit classifier across all 3 meters and all digit positions

Target design:
1. align a frame to a canonical crop
2. split into digit cells
3. normalize each digit cell
4. extract HOG features
5. classify each digit with scikit-learn
6. combine digits into a reading
7. apply temporal smoothing and monotonicity checks
8. persist uncertain frames for later review

---

## 2. Non-goals for v1

Do not implement these in v1:
- CNNs
- full free-form OCR
- complex adaptive segmentation across the full meter line
- per-position models
- online retraining inside the watch process
- database storage
- web UI

---

## 3. Recommended repository layout

```text
water-meter-ocr/
├─ pyproject.toml
├─ README.md
├─ configs/
│  ├─ meters.yaml
│  └─ defaults.yaml
├─ src/
│  └─ meterocr/
│     ├─ __init__.py
│     ├─ types.py
│     ├─ meter_config.py
│     ├─ align.py
│     ├─ segment.py
│     ├─ normalize.py
│     ├─ features.py
│     ├─ dataset.py
│     ├─ labeling.py
│     ├─ model_io.py
│     ├─ train.py
│     ├─ evaluate.py
│     ├─ predict.py
│     ├─ temporal.py
│     ├─ review.py
│     ├─ utils.py
│     └─ cli.py
├─ data/
│  ├─ raw/
│  ├─ aligned/
│  ├─ cells/
│  ├─ normalized/
│  ├─ labels/
│  │  ├─ frames.csv
│  │  ├─ samples.csv
│  │  └─ review_queue.csv
│  ├─ predictions/
│  └─ reports/
├─ models/
│  └─ digit_clf.joblib
└─ tests/
   ├─ test_align.py
   ├─ test_segment.py
   ├─ test_normalize.py
   ├─ test_features.py
   ├─ test_train.py
   ├─ test_predict.py
   └─ test_temporal.py
```

---

## 4. Coding rules

Implementation expectations:
- use Python type hints everywhere
- all public functions must have docstrings
- prefer dataclasses for internal structured types
- avoid global mutable state
- all paths should use `pathlib.Path`
- all image arrays should be BGR for OpenCV I/O and grayscale only after explicit conversion
- persist enough debug output to diagnose failures
- keep preprocessing deterministic
- expose behavior through CLI subcommands
- return structured results rather than ad-hoc tuples when more than 2 values are returned

---

## 5. Core types

Create these in `src/meterocr/types.py`.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


ImageU8 = NDArray[np.uint8]
FeatureVectorF32 = NDArray[np.float32]


@dataclass(frozen=True)
class DigitBox:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class MeterConfig:
    meter_id: str
    aligned_width: int
    aligned_height: int
    digit_boxes: list[DigitBox]
    threshold_mode: Literal["otsu", "adaptive"]
    invert_binary: bool
    crop_source_box: DigitBox | None = None
    perspective_src_points: list[tuple[float, float]] | None = None
    alignment_reference_path: Path | None = None
    max_translation_px: int = 0
    inner_pad_x: int = 0
    inner_pad_y: int = 0


@dataclass(frozen=True)
class HOGConfig:
    image_width: int = 40
    image_height: int = 64
    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (4, 4)
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = False


@dataclass(frozen=True)
class NormalizationConfig:
    blur_kernel: int = 3
    min_component_area: int = 8
    bbox_margin_px: int = 2
    output_width: int = 40
    output_height: int = 64
    foreground: Literal["white", "black"] = "white"


@dataclass(frozen=True)
class TrainingConfig:
    model_type: Literal["knn", "linear_svc"]
    test_size: float = 0.2
    group_by: Literal["frame", "date", "meter"] = "frame"
    random_state: int = 42
    knn_neighbors: int = 3
    knn_weights: Literal["uniform", "distance"] = "distance"
    svc_c: float = 1.0
    use_standard_scaler: bool = True


@dataclass(frozen=True)
class FrameMeta:
    frame_id: str
    meter_id: str
    timestamp: datetime
    source_path: Path


@dataclass(frozen=True)
class DigitCell:
    frame_id: str
    meter_id: str
    position: int
    bbox: DigitBox
    image_bgr: ImageU8


@dataclass(frozen=True)
class NormalizationResult:
    normalized: ImageU8
    gray: ImageU8
    binary: ImageU8
    digit_bbox_xywh: tuple[int, int, int, int] | None
    success: bool
    reason: str


@dataclass(frozen=True)
class DigitSample:
    frame_id: str
    meter_id: str
    timestamp: datetime
    position: int
    digit_label: int
    full_reading: str
    raw_cell_path: Path
    normalized_path: Path
    quality: Literal["ok", "uncertain", "rejected"] = "ok"


@dataclass(frozen=True)
class DigitPrediction:
    position: int
    digit: int
    confidence: float
    margin: float
    top2_digits: tuple[int, int] | None = None
    top2_scores: tuple[float, float] | None = None


@dataclass(frozen=True)
class PredictionResult:
    frame_id: str
    meter_id: str
    timestamp: datetime
    raw_reading: str
    digits: list[DigitPrediction]
    min_confidence: float
    mean_confidence: float


@dataclass(frozen=True)
class StableReadingResult:
    frame_id: str
    meter_id: str
    timestamp: datetime
    accepted: bool
    raw_reading: str
    stable_reading: str
    reason: str


@dataclass
class MeterState:
    meter_id: str
    last_stable_reading: str | None = None
    last_stable_timestamp: datetime | None = None
    recent_predictions: list[PredictionResult] = field(default_factory=list)
```

Notes:
- `confidence` and `margin` may be identical in v1 for `LinearSVC`
- `top2_*` may be `None` in v1 if not implemented immediately

---

## 6. Configuration files

### 6.1 `configs/meters.yaml`

Use YAML with one entry per meter.

Example shape:

```yaml
meters:
  - meter_id: M1
    aligned_width: 500
    aligned_height: 120
    threshold_mode: otsu
    invert_binary: true
    crop_source_box: { x: 120, y: 80, w: 500, h: 120 }
    max_translation_px: 4
    inner_pad_x: 2
    inner_pad_y: 2
    digit_boxes:
      - { x: 0, y: 0, w: 92, h: 120 }
      - { x: 96, y: 0, w: 92, h: 120 }
      - { x: 192, y: 0, w: 92, h: 120 }
      - { x: 288, y: 0, w: 92, h: 120 }
      - { x: 384, y: 0, w: 92, h: 120 }

  - meter_id: M2
    aligned_width: 500
    aligned_height: 120
    threshold_mode: otsu
    invert_binary: true
    crop_source_box: { x: 140, y: 90, w: 500, h: 120 }
    max_translation_px: 4
    inner_pad_x: 2
    inner_pad_y: 2
    digit_boxes:
      - { x: 0, y: 0, w: 92, h: 120 }
      - { x: 96, y: 0, w: 92, h: 120 }
      - { x: 192, y: 0, w: 92, h: 120 }
      - { x: 288, y: 0, w: 92, h: 120 }
      - { x: 384, y: 0, w: 92, h: 120 }
```

### 6.2 `configs/defaults.yaml`

```yaml
hog:
  image_width: 40
  image_height: 64
  orientations: 9
  pixels_per_cell: [4, 4]
  cells_per_block: [2, 2]
  block_norm: L2-Hys
  transform_sqrt: false

normalization:
  blur_kernel: 3
  min_component_area: 8
  bbox_margin_px: 2
  output_width: 40
  output_height: 64
  foreground: white

training:
  model_type: linear_svc
  test_size: 0.2
  group_by: frame
  random_state: 42
  knn_neighbors: 3
  knn_weights: distance
  svc_c: 1.0
  use_standard_scaler: true
```

---

## 7. Module responsibilities and exact function signatures

### 7.1 `meter_config.py`

Responsibilities:
- load YAML config
- validate config
- expose meter lookup by id

Required public API:

```python
from pathlib import Path

from meterocr.types import HOGConfig, MeterConfig, NormalizationConfig, TrainingConfig


def load_meter_configs(path: Path) -> dict[str, MeterConfig]:
    ...


def load_default_configs(path: Path) -> tuple[HOGConfig, NormalizationConfig, TrainingConfig]:
    ...


def get_meter_config(configs: dict[str, MeterConfig], meter_id: str) -> MeterConfig:
    ...
```

Implementation notes:
- raise `KeyError` for unknown `meter_id`
- validate that `digit_boxes` are inside the aligned dimensions
- validate either `crop_source_box` or `perspective_src_points` may exist, or neither if input is already aligned

---

### 7.2 `align.py`

Responsibilities:
- crop or perspective-transform the source image into an aligned meter image
- optionally apply small translation correction against a reference

Required public API:

```python
from meterocr.types import ImageU8, MeterConfig


def align_meter(frame_bgr: ImageU8, config: MeterConfig) -> ImageU8:
    ...


def crop_meter(frame_bgr: ImageU8, config: MeterConfig) -> ImageU8:
    ...


def warp_meter(frame_bgr: ImageU8, config: MeterConfig) -> ImageU8:
    ...


def estimate_translation(reference_gray: ImageU8, candidate_gray: ImageU8, max_shift_px: int) -> tuple[int, int]:
    ...


def apply_translation(image: ImageU8, dx: int, dy: int) -> ImageU8:
    ...
```

Implementation notes:
- `align_meter` chooses crop or warp based on config
- if `alignment_reference_path` exists and `max_translation_px > 0`, apply bounded translation after crop/warp
- use grayscale normalized cross-correlation or template matching for translation only
- do not implement arbitrary rotation/skew correction in this translation step

---

### 7.3 `segment.py`

Responsibilities:
- extract fixed-position digit cells from an aligned meter image

Required public API:

```python
from meterocr.types import DigitCell, FrameMeta, ImageU8, MeterConfig


def extract_digit_cells(aligned_meter_bgr: ImageU8, config: MeterConfig, frame_meta: FrameMeta) -> list[DigitCell]:
    ...


def crop_digit_cell(aligned_meter_bgr: ImageU8, x: int, y: int, w: int, h: int) -> ImageU8:
    ...
```

Implementation notes:
- apply `inner_pad_x` / `inner_pad_y` before extracting the final cell
- preserve BGR for downstream debug output
- positions must be zero-based and ordered left-to-right

---

### 7.4 `normalize.py`

Responsibilities:
- transform a raw cell into a canonical image suitable for HOG classification

Required public API:

```python
from meterocr.types import ImageU8, NormalizationConfig, NormalizationResult


def normalize_digit(cell_bgr: ImageU8, threshold_mode: str, invert_binary: bool, cfg: NormalizationConfig) -> NormalizationResult:
    ...


def to_gray(cell_bgr: ImageU8) -> ImageU8:
    ...


def threshold_digit(gray: ImageU8, mode: str, invert_binary: bool) -> ImageU8:
    ...


def remove_small_components(binary_img: ImageU8, min_area: int) -> ImageU8:
    ...


def find_main_digit_bbox(binary_img: ImageU8, min_area: int) -> tuple[int, int, int, int] | None:
    ...


def center_and_resize(binary_digit: ImageU8, output_size: tuple[int, int]) -> ImageU8:
    ...
```

Implementation notes:
- output must be deterministic
- if no digit blob is found, return `success=False` and a clear `reason`
- in v1, normalization may be binary-only; keep `gray` in the result for debug
- prefer digit foreground white on black unless testing proves the opposite is better
- `center_and_resize` should preserve aspect ratio and pad to the target size

---

### 7.5 `features.py`

Responsibilities:
- convert normalized digit images into HOG feature vectors

Required public API:

```python
import numpy as np

from meterocr.types import FeatureVectorF32, HOGConfig, ImageU8


def extract_hog_features(normalized_img: ImageU8, cfg: HOGConfig) -> FeatureVectorF32:
    ...


def extract_feature_matrix(images: list[ImageU8], cfg: HOGConfig) -> np.ndarray:
    ...
```

Implementation notes:
- return `np.float32`
- normalized images should already be the correct size
- `extract_feature_matrix` should stack row-wise into shape `(n_samples, n_features)`

---

### 7.6 `dataset.py`

Responsibilities:
- append frame labels
- derive digit samples from a full reading label
- load samples for training/evaluation
- manage CSV persistence

Required public API:

```python
from datetime import datetime
from pathlib import Path

import pandas as pd

from meterocr.types import DigitCell, DigitSample, FrameMeta, ImageU8, NormalizationResult


def append_frame_label(
    labels_csv: Path,
    frame_id: str,
    meter_id: str,
    timestamp: datetime,
    source_path: Path,
    full_reading: str,
) -> None:
    ...


def derive_digit_samples(
    frame_meta: FrameMeta,
    full_reading: str,
    cells: list[DigitCell],
    normalized_results: list[NormalizationResult],
    raw_cell_dir: Path,
    normalized_dir: Path,
) -> list[DigitSample]:
    ...


def append_digit_samples(samples_csv: Path, samples: list[DigitSample]) -> None:
    ...


def load_samples(samples_csv: Path) -> pd.DataFrame:
    ...


def load_frame_labels(labels_csv: Path) -> pd.DataFrame:
    ...
```

Implementation notes:
- `derive_digit_samples` must validate that `len(full_reading) == len(cells)`
- write raw cell and normalized images to disk before emitting rows
- store paths relative to project root if practical
- use append-safe CSV writing
- tolerate missing CSV by creating it with headers

Expected `frames.csv` columns:
- `frame_id`
- `timestamp`
- `meter_id`
- `source_path`
- `full_reading`

Expected `samples.csv` columns:
- `frame_id`
- `timestamp`
- `meter_id`
- `position`
- `digit_label`
- `full_reading`
- `raw_cell_path`
- `normalized_path`
- `quality`

---

### 7.7 `labeling.py`

Responsibilities:
- take a labeled frame and produce derived per-digit samples in one step

Required public API:

```python
from pathlib import Path

from meterocr.types import FrameMeta, MeterConfig, NormalizationConfig


def label_frame(
    image_path: Path,
    meter_config: MeterConfig,
    normalization_cfg: NormalizationConfig,
    full_reading: str,
    frames_csv: Path,
    samples_csv: Path,
    raw_cell_dir: Path,
    normalized_dir: Path,
) -> None:
    ...
```

Implementation notes:
- this is a convenience wrapper that calls align, segment, normalize, dataset functions
- generate `frame_id` from timestamp + meter_id or from filename stem
- reject labels with non-digit characters in v1

---

### 7.8 `model_io.py`

Responsibilities:
- save and load trained model bundle

Required public API:

```python
from pathlib import Path
from typing import Any

from meterocr.types import HOGConfig, NormalizationConfig, TrainingConfig


def save_model_bundle(
    model_path: Path,
    classifier: Any,
    hog_cfg: HOGConfig,
    normalization_cfg: NormalizationConfig,
    training_cfg: TrainingConfig,
    class_labels: list[int],
) -> None:
    ...


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    ...
```

Model bundle must contain:
- classifier
- scaler if used
- HOG config
- normalization config
- training config
- class labels
- implementation version string

Use `joblib.dump` / `joblib.load`.

---

### 7.9 `train.py`

Responsibilities:
- load dataset
- extract features
- fit model
- save bundle
- return evaluation data

Required public API:

```python
from pathlib import Path
from typing import Any

import pandas as pd

from meterocr.types import HOGConfig, NormalizationConfig, TrainingConfig


def build_training_dataframe(samples_csv: Path) -> pd.DataFrame:
    ...


def split_train_validation(df: pd.DataFrame, cfg: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...


def train_knn(
    train_df: pd.DataFrame,
    hog_cfg: HOGConfig,
    normalization_cfg: NormalizationConfig,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    ...


def train_linear_svc(
    train_df: pd.DataFrame,
    hog_cfg: HOGConfig,
    normalization_cfg: NormalizationConfig,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    ...


def train_and_save_model(
    samples_csv: Path,
    model_path: Path,
    hog_cfg: HOGConfig,
    normalization_cfg: NormalizationConfig,
    training_cfg: TrainingConfig,
) -> dict[str, Any]:
    ...
```

Expected return dict from training helpers:
- `classifier`
- `scaler`
- `train_count`
- `feature_dim`
- `class_labels`

Implementation notes:
- read normalized images from `normalized_path`
- optional scaler is allowed before `LinearSVC`
- group split should avoid frame leakage
- `train_and_save_model` should dispatch based on `training_cfg.model_type`

---

### 7.10 `evaluate.py`

Responsibilities:
- evaluate a model on a validation set
- save confusion matrix and misclassified samples

Required public API:

```python
from pathlib import Path
from typing import Any

import pandas as pd


def evaluate_digit_classifier(
    val_df: pd.DataFrame,
    classifier: Any,
    scaler: Any | None,
    reports_dir: Path,
) -> dict[str, Any]:
    ...


def save_confusion_report(
    y_true: list[int],
    y_pred: list[int],
    reports_dir: Path,
    label: str,
) -> None:
    ...


def save_misclassified_examples(
    val_df: pd.DataFrame,
    y_pred: list[int],
    reports_dir: Path,
    limit: int = 100,
) -> None:
    ...
```

Metrics to compute:
- digit accuracy
- confusion matrix
- per-meter accuracy
- per-position accuracy

Implementation notes:
- use validation dataframe rows to recover `meter_id`, `position`, `normalized_path`
- persist CSV summaries to `reports_dir`

---

### 7.11 `predict.py`

Responsibilities:
- run full inference on one frame
- emit per-digit and full-reading predictions

Required public API:

```python
from datetime import datetime
from pathlib import Path
from typing import Any

from meterocr.types import MeterConfig, PredictionResult


def predict_meter_reading(
    image_path: Path,
    meter_config: MeterConfig,
    model_bundle: dict[str, Any],
    timestamp: datetime | None = None,
) -> PredictionResult:
    ...


def predict_meter_reading_from_array(
    frame_bgr,
    frame_id: str,
    timestamp: datetime,
    meter_config: MeterConfig,
    model_bundle: dict[str, Any],
) -> PredictionResult:
    ...


def classify_normalized_digit(
    normalized_img,
    classifier: Any,
    scaler: Any | None,
    hog_cfg,
) -> tuple[int, float, float]:
    ...


def combine_digits_to_reading(digits: list[int]) -> str:
    ...
```

Implementation notes:
- `predict_meter_reading` is file-based convenience wrapper
- `classify_normalized_digit` returns `(digit, confidence, margin)`
- for `LinearSVC`, use decision function margin as confidence proxy
- for multi-class decision scores, record the best class and optionally second best

---

### 7.12 `temporal.py`

Responsibilities:
- maintain recent predictions per meter
- accept, reject, or defer readings based on temporal rules

Required public API:

```python
from datetime import timedelta

from meterocr.types import MeterState, PredictionResult, StableReadingResult


def reading_to_int(reading: str) -> int:
    ...


def update_meter_state(
    state: MeterState,
    prediction: PredictionResult,
    min_confidence_threshold: float,
    max_delta_per_hour: float | None = None,
    rolling_window_size: int = 5,
) -> StableReadingResult:
    ...


def is_monotonic(last_reading: str | None, new_reading: str) -> bool:
    ...


def is_plausible_delta(
    last_reading: str | None,
    last_timestamp,
    new_reading: str,
    new_timestamp,
    max_delta_per_hour: float | None,
) -> bool:
    ...


def prune_recent_predictions(state: MeterState, max_items: int) -> None:
    ...
```

Required behavior:
- reject or hold readings that go backwards
- reject or hold large implausible jumps if `max_delta_per_hour` is set
- if confidence is below threshold, return `accepted=False` and keep previous stable reading
- if accepted, update `last_stable_reading` and `last_stable_timestamp`
- `recent_predictions` should maintain a small rolling window

Implementation note:
- v1 may simply return the new reading if monotonic + plausible + confidence above threshold
- rolling vote can be added after baseline works

---

### 7.13 `review.py`

Responsibilities:
- collect uncertain or rejected predictions for later manual review

Required public API:

```python
from pathlib import Path

from meterocr.types import PredictionResult, StableReadingResult


def append_review_item(
    review_csv: Path,
    prediction: PredictionResult,
    stable: StableReadingResult,
    image_path: Path | None,
) -> None:
    ...
```

Expected CSV columns:
- `frame_id`
- `timestamp`
- `meter_id`
- `raw_reading`
- `stable_reading`
- `accepted`
- `reason`
- `min_confidence`
- `mean_confidence`
- `image_path`

---

### 7.14 `cli.py`

Responsibilities:
- provide an ergonomic CLI for labeling, training, evaluating, predicting, and watching streams

Required commands:
- `label-frame`
- `train`
- `evaluate`
- `predict`
- `watch`

Minimum command behavior:

```text
meterocr label-frame --meter M1 --image path/to/frame.png --reading 03506
meterocr train --samples data/labels/samples.csv --model models/digit_clf.joblib
meterocr evaluate --samples data/labels/samples.csv --model models/digit_clf.joblib --reports data/reports
meterocr predict --meter M1 --image path/to/frame.png --model models/digit_clf.joblib
meterocr watch --meter M1 --source rtsp://... --model models/digit_clf.joblib
```

`watch` requirements for v1:
- periodic frame grab, not real-time high-throughput streaming
- one meter per process is acceptable
- persist predictions to CSV
- optionally save uncertain frames for review

---

## 8. Data flow

### 8.1 Bootstrapping labeled data
1. manually capture frame image
2. run `label-frame`
3. CLI:
   - loads configs
   - aligns meter
   - segments digit cells
   - normalizes each digit
   - writes raw and normalized images
   - appends one row to `frames.csv`
   - appends one row per digit to `samples.csv`

### 8.2 Training
1. run `train`
2. CLI loads `samples.csv`
3. training code loads normalized images and labels
4. extracts HOG features
5. splits by frame groups
6. trains classifier
7. saves joblib bundle

### 8.3 Prediction
1. run `predict` or `watch`
2. load model bundle + meter config
3. align -> segment -> normalize -> HOG -> classify
4. combine to reading
5. pass through temporal layer
6. log accepted or uncertain output
7. append bad cases to review queue

---

## 9. Model details

### v1 baseline
Implement both:
- kNN baseline
- LinearSVC production candidate

Default production path:
- HOG
- StandardScaler
- LinearSVC(C=1.0)

Expected classifier semantics:
- classes are integers 0..9
- model predicts one digit per normalized image
- classifier operates on pooled samples from all meters and positions

### Confidence
For `LinearSVC`:
- use `decision_function`
- let winning class score be `margin`
- map to `confidence = margin` in v1
- exact calibration is not required initially

For `kNN`:
- confidence can be the fraction of weighted votes or inverse distance heuristic
- acceptable if crude

---

## 10. Acceptance criteria

### Functional acceptance
A v1 implementation is acceptable if:
- labeling one frame produces the expected rows in `frames.csv` and `samples.csv`
- training completes without manual edits
- prediction returns a structured `PredictionResult`
- temporal filter rejects or holds obviously bad readings
- review queue is populated for uncertain frames

### Quality acceptance
A good first version should:
- correctly read most clean frames
- never silently accept a backwards reading
- be easy to improve by adding more labeled data
- save enough debug artifacts to diagnose errors

---

## 11. Tests to implement

Minimum tests:
- `test_align.py`
  - crop path returns expected size
- `test_segment.py`
  - extracted cell count equals configured digit count
- `test_normalize.py`
  - normalized image matches configured output size
  - missing-blob case returns `success=False`
- `test_features.py`
  - HOG feature vector has stable shape
- `test_train.py`
  - tiny synthetic dataset can train and save bundle
- `test_predict.py`
  - mocked classifier returns expected reading structure
- `test_temporal.py`
  - backwards reading is rejected
  - low-confidence reading is held
  - monotonic plausible reading is accepted

---

## 12. Implementation order

Implement in this order.

### Step 1
Create:
- `types.py`
- `meter_config.py`
- `align.py`
- `segment.py`

Goal:
- be able to load a frame and extract fixed digit cells

### Step 2
Create:
- `normalize.py`

Goal:
- save normalized images and visually confirm shape consistency

### Step 3
Create:
- `dataset.py`
- `labeling.py`
- `cli.py` with `label-frame`

Goal:
- bootstrap dataset collection immediately

### Step 4
Create:
- `features.py`
- `train.py`
- `model_io.py`

Goal:
- train kNN baseline, then LinearSVC

### Step 5
Create:
- `predict.py`
- `evaluate.py`
- `cli.py` commands `predict`, `train`, `evaluate`

Goal:
- full offline inference loop

### Step 6
Create:
- `temporal.py`
- `review.py`
- `cli.py` command `watch`

Goal:
- long-running meter monitoring with uncertainty handling

---

## 13. Suggested implementation details worth keeping simple

Keep these simple in v1:
- alignment correction: translation only
- segmentation: fixed boxes only
- thresholding: Otsu first
- normalization: biggest connected component
- training split: grouped by frame_id
- temporal logic: monotonic + confidence + plausible delta only

Do not over-engineer the first version.

---

## 14. Practical pitfalls to avoid

- leaking near-identical frames across train/validation split
- changing normalization logic between training and inference
- silently accepting empty or failed normalization results
- mixing grayscale polarity between samples
- using different target sizes for normalized images vs HOG config
- forgetting to persist debug artifacts for uncertain cases

---

## 15. Nice-to-have but optional after v1 works

Only after the basic system works:
- small translation/blur augmentation at training time
- top-2 candidate reporting per digit
- calibration of SVC margins into probabilities
- per-position fallback heuristics for rare trouble spots
- automatic reference alignment update tools
- simple review CLI for relabeling uncertain frames

---

## 16. Final instruction to the implementing agent

Build the system exactly around:
- fixed geometry
- deterministic normalization
- pooled HOG digit classification
- temporal monotonicity checks

Prefer code that is boring, testable, and inspectable over clever code.
The system should fail safely by marking frames uncertain rather than inventing readings.
