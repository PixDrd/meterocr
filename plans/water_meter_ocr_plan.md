# Water Meter OCR Plan (scikit-image + scikit-learn)

## Goal
Implement a robust OCR pipeline for 3 indoor water meters with fairly constant lighting, using:
- OpenCV for image I/O, alignment, ROI extraction, and thresholding
- scikit-image for feature extraction (mainly HOG)
- scikit-learn for digit classification and confidence scoring

Constraints:
- No custom CNN training
- Samples must be collected over time from real meter readings
- Samples should be shared across digit positions and across the 3 meters
- Input is assumed to already be roughly cropped/perspective-corrected to the meter window, or can be aligned to a fixed reference first

---

## High-level strategy
Do **not** treat the meter as free-form OCR text.
Treat it as a row of fixed digit wheels.

Pipeline:
1. Align the full meter crop to a canonical rectangle
2. Split into fixed digit cells
3. Normalize each digit cell aggressively so shape matters more than position/lighting
4. Extract HOG features from each normalized digit image
5. Train a pooled digit classifier across all meters and all positions
6. Read full meter values per frame
7. Apply temporal consistency rules so the final reading is monotonic and stable

Indoor/fairly constant lighting simplifies the problem enough that this should be reliable without deep learning.

---

## Recommended architecture

### Modules
- `meter_config.py`
  - per-meter geometry/config
- `align.py`
  - optional alignment of raw crop to canonical crop
- `segment.py`
  - split meter into digit cells
- `normalize.py`
  - normalize a single digit image to a canonical form
- `features.py`
  - HOG feature extraction
- `dataset.py`
  - sample storage, metadata, dataset loading
- `train.py`
  - model training / validation / export
- `predict.py`
  - per-frame inference
- `temporal.py`
  - smoothing and monotonicity logic
- `cli.py`
  - command-line entry points

### Persisted artifacts
- `data/raw/`
  - original full meter images
- `data/crops/`
  - canonical aligned meter crops
- `data/cells/`
  - extracted digit cells
- `data/normalized/`
  - normalized digit images used for training/debug
- `data/labels.csv`
  - one row per frame, contains full reading + metadata
- `models/digit_clf.joblib`
- `models/preprocessing_config.json`
- `reports/`
  - confusion matrices, validation reports, bad samples

---

## Assumptions
- Each meter has the same basic font/wheel style
- Lighting is mostly stable
- Camera position is mostly fixed, with only small drift
- Meter values are monotonic non-decreasing
- The integer wheels are the primary target; fractional/red wheels can be ignored initially unless needed

---

## Phase 1: meter alignment

### Objective
Convert each incoming frame into a canonical meter image with the same size and orientation.

### Implementation
Per meter, store either:
- a fixed crop rectangle if the camera is locked well enough, or
- four reference corners / homography if a perspective warp is still needed

### Recommended approach
Start simple:
- define a per-meter crop rectangle / warp once
- save aligned meter crop to a fixed size, e.g. `W x H`

Optional refinement:
- use template matching or feature matching against a reference frame to correct small x/y drift before splitting digits

### Deliverable
Function:
- `align_meter(frame, meter_id) -> aligned_meter_bgr`

---

## Phase 2: digit segmentation

### Objective
Split the aligned meter image into individual digit cells.

### Recommended approach
Use fixed cell boundaries based on known geometry.

For each meter config store:
- full meter ROI size
- number of digits
- x offsets for each digit cell, or one equal-width scheme if spacing is uniform
- optional small inner padding per cell to remove borders/separators

Because lighting is stable, fixed segmentation is preferable to clever dynamic segmentation.

### Optional refinement
Allow a very small per-frame x adjustment using vertical projection / edge energy if needed, but keep this bounded. Do not build a free-form segmentation system.

### Deliverable
Function:
- `extract_digit_cells(aligned_meter, meter_id) -> list[DigitCell]`

Where `DigitCell` includes:
- image
- meter_id
- position_index
- timestamp / frame_id

---

## Phase 3: digit normalization

### Objective
Transform each raw digit cell into a canonical small binary/grayscale image so samples can be pooled across meters and positions.

### Why this matters
This is the most important part. Good normalization reduces dependency on exact placement and lighting and makes pooled training viable.

### Normalization pipeline
For each digit cell:
1. Convert to grayscale
2. Apply light denoise if needed (`GaussianBlur` 3x3)
3. Apply thresholding
   - first try Otsu
   - keep adaptive threshold as fallback only if needed
4. Ensure digit is foreground on dark background or vice versa, but keep it consistent across all samples
5. Remove tiny connected components / dust
6. Find the main digit blob using connected components or bounding box of foreground pixels
7. Crop tightly around the digit blob with a small margin
8. Center the cropped digit into a fixed canvas
9. Resize to a fixed final size, e.g. `32x48` or `40x64`
10. Optionally keep both:
   - binary normalized image
   - grayscale normalized image

### Important rules
- The same normalization code must be used for both training and inference
- Save debug images for failed or uncertain reads
- Keep preprocessing deterministic

### Deliverable
Function:
- `normalize_digit(cell_img) -> normalized_img`

---

## Phase 4: sample collection and labeling

### Core idea
Label entire meter readings over time, then derive per-digit labels automatically from the full reading.

This is how to get training data without manually forcing specific digits to appear.

### Data collection strategy
For each new frame:
- capture image
- align meter
- split into cells
- store frame metadata
- manually enter the full visible reading for that frame during the bootstrap phase

Then for each position in the reading:
- assign the corresponding digit label to the corresponding cell
- add the sample to the pooled dataset

### Why pooling is acceptable
A `3` from meter A, position 1 should be usable as training data for meter B, position 4, as long as normalization is good.

### Metadata to store per sample
For each digit sample store:
- sample path
- normalized sample path
- label digit `0-9`
- full reading string
- meter id
- digit position index
- timestamp
- frame id
- optional quality flags

### File format
Use a CSV or parquet file, e.g. `data/samples.csv`.

Suggested columns:
- `frame_id`
- `timestamp`
- `meter_id`
- `position`
- `digit_label`
- `full_reading`
- `raw_cell_path`
- `normalized_path`
- `quality`

### Bootstrap target
Initial target:
- at least 10-20 samples per digit overall across all meters
- more is better, but perfect position coverage is not required

---

## Phase 5: feature extraction

### Recommended feature
Use **HOG** from `skimage.feature.hog`.

Why:
- easy to implement
- works well on centered character shapes
- much more tolerant than raw template matching
- no neural training required

### Suggested HOG parameters
Start with something like:
- `orientations=9`
- `pixels_per_cell=(4, 4)` or `(8, 8)` depending on final image size
- `cells_per_block=(2, 2)`
- `block_norm='L2-Hys'`

Exact values can be tuned after initial validation.

### Optional extra features
Only if needed later:
- aspect ratio of digit bounding box
- foreground pixel ratio
- horizontal/vertical projection histograms

Do **not** add complexity early unless HOG alone is insufficient.

### Deliverable
Function:
- `extract_features(normalized_img) -> np.ndarray`

---

## Phase 6: model training

### Baseline model to implement first
Use one simple pooled classifier over digits `0-9`.

Recommended order:
1. `KNeighborsClassifier(weights='distance')` as an easy sanity-check baseline
2. `LinearSVC` as the likely production model
3. Optional: wrap SVC in calibration if probability-style confidence is needed later

### Recommended first production choice
`LinearSVC`

Reason:
- simple
- fast
- often works very well on HOG features
- robust enough for digit classification

### Training split
Do not randomly split individual digit samples only.
Prefer splitting by frame or timestamp blocks so near-duplicate frames do not leak between train and validation.

Better still:
- train on a subset of dates/frames
- validate on later unseen dates/frames

This gives a much more honest estimate.

### Metrics to report
- per-digit accuracy
- confusion matrix
- per-meter accuracy
- per-position accuracy
- full-reading exact match rate before temporal smoothing
- full-reading exact match rate after temporal smoothing

### Deliverables
Functions/scripts:
- `train_digit_classifier(samples_csv, out_model)`
- `evaluate_digit_classifier(...)`

Persist:
- classifier with `joblib`
- any scaler/preprocessor if used
- HOG config
- normalization config

---

## Phase 7: inference

### Per-frame inference
For each new frame:
1. align full meter image
2. extract digit cells
3. normalize each cell
4. compute HOG features
5. predict digit class per cell
6. also return confidence / margin per digit
7. combine into a full reading string

### Confidence handling
For `LinearSVC`, use decision margin as a confidence proxy.
For `kNN`, use neighbor vote/distance as a crude confidence.

Store:
- predicted digit
- confidence per digit
- full reading
- average/min confidence for the frame

### Deliverable
Function:
- `predict_meter_reading(frame, meter_id) -> PredictionResult`

`PredictionResult` should include:
- raw predicted reading
- per-digit predictions
- per-digit confidences
- normalized digit images for debug

---

## Phase 8: temporal consistency layer

### Objective
Exploit the fact that water meter values are monotonic and usually change slowly.

This is essential for making the system robust during borderline frames.

### Rules to implement
For each meter independently:

1. **Monotonic non-decreasing**
   - reject or down-rank readings lower than the last accepted reading

2. **Max plausible delta per interval**
   - if the frame cadence is known, reject implausibly large jumps unless repeated consistently

3. **Short-window voting**
   - keep a rolling window of the last `N` predictions
   - use majority vote or best-confidence stable reading

4. **Uncertain carry transitions**
   - when one digit is low-confidence and adjacent digits suggest a carry/rollover, prefer holding the previous stable reading until subsequent frames resolve it

5. **Confidence thresholding**
   - if minimum digit confidence is below threshold, mark frame as uncertain instead of forcing acceptance

### Practical output states
Each frame should end up as one of:
- accepted reading
- uncertain / hold previous stable value
- rejected as implausible

### Deliverable
Function:
- `update_meter_state(meter_id, prediction, timestamp) -> StableReadingResult`

---

## Phase 9: active learning / review loop

### Objective
Minimize manual labeling effort after bootstrap.

### Workflow
Once the first model exists:
- run it on new frames
- auto-accept high-confidence, temporally consistent samples
- surface only uncertain or inconsistent frames for review
- after review, add those corrected samples back into the dataset
- retrain periodically

### Frames to flag for review
- low-confidence digit(s)
- reading goes backwards
- implausible jump
- disagreement between raw prediction and temporal smoother
- normalization failure (no clear foreground blob)

### Benefit
This gives a sustainable long-term improvement path without ever needing to stage artificial digit values.

---

## Handling incomplete digit coverage

### Problem
You may never observe all digits equally in the leftmost positions.

### Plan
That is acceptable.

Mitigation strategy:
1. pool samples across meters and positions
2. normalize digits hard enough that position-specific differences mostly disappear
3. collect over time rather than trying to force exhaustive coverage
4. use temporal logic to handle rare ambiguous cases

### Optional fallback if needed later
If one specific position behaves differently, add a lightweight fallback:
- global pooled classifier first
- optional per-position calibration or tie-break logic only for problematic positions

Do not start with separate per-position classifiers.

---

## Optional data augmentation
Because lighting is fairly stable, augmentation can stay modest.

Useful augmentations on normalized images:
- small x/y translations (1-2 px)
- tiny scale variation
- slight blur
- mild contrast variation before thresholding

Avoid heavy augmentation that makes digits unrealistic.

Start without augmentation. Add it only if validation shows sensitivity to small shifts.

---

## Recommended implementation order

### Step 1
Implement fixed alignment + fixed digit segmentation.

### Step 2
Implement deterministic digit normalization and save debug outputs.

### Step 3
Implement manual labeling from full reading strings and build `samples.csv`.

### Step 4
Implement HOG feature extraction.

### Step 5
Train a baseline `KNeighborsClassifier` to verify the pipeline works end-to-end.

### Step 6
Train `LinearSVC` and compare results.

### Step 7
Implement per-frame confidence reporting.

### Step 8
Implement temporal consistency and stable accepted readings.

### Step 9
Implement active-learning review flow for uncertain frames.

---

## Suggested CLI commands

### Collect / preprocess
- `meterocr extract-crops --meter M1 --input ... --output ...`
- `meterocr extract-cells --meter M1 --input ... --output ...`
- `meterocr normalize-cells --samples data/samples.csv`

### Label bootstrap data
- `meterocr label-frame --frame ... --meter M1 --reading 03506`
- or batch import labels from CSV

### Train / evaluate
- `meterocr train --samples data/samples.csv --model models/digit_clf.joblib`
- `meterocr evaluate --samples data/samples.csv --model models/digit_clf.joblib`

### Predict
- `meterocr predict --meter M1 --image frame.png`
- `meterocr watch --meter M1 --rtsp ...`

### Review
- `meterocr review --only uncertain`

---

## Suggested data structures

### Meter config
```python
@dataclass
class MeterConfig:
    meter_id: str
    aligned_size: tuple[int, int]
    digit_boxes: list[tuple[int, int, int, int]]
    threshold_mode: str
```

### Prediction result
```python
@dataclass
class DigitPrediction:
    digit: int
    confidence: float
    position: int

@dataclass
class PredictionResult:
    meter_id: str
    timestamp: datetime
    reading: str
    digits: list[DigitPrediction]
    min_confidence: float
    mean_confidence: float
```

### Stable reading result
```python
@dataclass
class StableReadingResult:
    accepted: bool
    stable_reading: str
    raw_reading: str
    reason: str
```

---

## Validation checklist
The implementation should make it easy to answer:
- Which digits are most frequently confused?
- Which meter has the worst accuracy?
- Which digit positions are problematic?
- How often does normalization fail?
- How much does temporal smoothing improve the final reading accuracy?

Save representative failure cases automatically.

---

## Non-goals for v1
- No deep learning
- No free-form OCR over the whole string
- No complex dynamic segmentation
- No per-position/per-meter custom models unless clearly justified by validation

---

## Recommended v1 acceptance criteria
A good v1 should:
- correctly classify most isolated normalized digits
- produce stable monotonically increasing readings in normal operation
- reject or hold uncertain readings rather than committing obvious errors
- be easy to improve by adding more labeled frames over time

---

## Summary of the intended design
Use a **pooled digit classifier** trained on **normalized per-digit crops** from all meters and all positions.
Use **HOG + scikit-learn** for simple, explainable classification.
Use **temporal monotonicity rules** to turn decent per-frame predictions into reliable meter readings.

That is the best fit for the constraints: easy to implement, no CNN training, tolerant of incomplete sample coverage, and likely reliable indoors with stable lighting.
