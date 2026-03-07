# bokevatten2 — Water Meter OCR

Reads three indoor water meters from USB webcam images using classical computer vision.
No deep learning. No cloud APIs. Runs on a Raspberry Pi or any Linux box.

**Pipeline:** crop → segment digit cells → normalize → HOG features → LinearSVC → temporal filter → stable reading

---

## How it works

Each meter is photographed by a dedicated USB webcam from a fixed position.
A per-meter config defines exactly where the meter is in the frame and where each digit cell sits.
Frames are processed independently; a temporal monotonicity filter turns per-frame predictions into reliable readings.

```
frame → align to canonical crop
      → split into N digit cells (fixed geometry)
      → normalize each cell (grayscale → Otsu threshold → largest blob → center+resize)
      → HOG features
      → LinearSVC → digit 0–9 per cell
      → combine → reading string
      → temporal filter (monotonic, confidence, plausible delta)
      → stable accepted reading
```

The same digit classifier is shared across all three meters and all digit positions (pooled training).

---

## Requirements

- Python 3.11+
- OpenCV-compatible USB webcams (or test images for offline use)

Install dependencies:

```bash
pip install -e ".[dev]"
```

---

## Project layout

```
configs/
  meters.yaml       # per-meter geometry (crop box, digit cell positions)
  defaults.yaml     # HOG, normalization, training hyperparameters

src/meterocr/
  types.py          # dataclasses
  meter_config.py   # YAML loading
  align.py          # crop / perspective warp / translation correction
  segment.py        # extract digit cells from aligned meter image
  normalize.py      # grayscale → binary → centered digit image
  features.py       # HOG feature extraction
  dataset.py        # CSV persistence for frames and samples
  labeling.py       # label a frame and derive per-digit samples
  model_io.py       # save/load joblib model bundle
  train.py          # kNN and LinearSVC training
  evaluate.py       # confusion matrix and accuracy reports
  predict.py        # per-frame inference
  temporal.py       # monotonicity and confidence filtering
  review.py         # queue uncertain frames for review
  capture.py        # USB webcam and offline test image acquisition
  cli.py            # command-line entry points

data/
  raw/              # original captured frames (gitignored)
  aligned/          # canonical aligned meter crops (gitignored)
  cells/            # extracted digit cells (gitignored)
  normalized/       # normalized digit images used for training (gitignored)
  labels/
    frames.csv      # one row per labeled frame
    samples.csv     # one row per labeled digit
    review_queue.csv
  predictions/      # watch output
  reports/          # confusion matrices, per-meter accuracy
  test_images/      # offline test images — M1/, M2/, M3/ subdirectories

models/
  digit_clf.joblib  # trained model bundle (gitignored)
```

---

## Initial setup: configure your meters

Edit `configs/meters.yaml` to match your actual camera positions.

For each meter you need:

- `crop_source_box` — the rectangle in the raw camera frame that contains the meter display (`x, y, w, h` in pixels)
- `aligned_width` / `aligned_height` — the canonical size you want to work with
- `digit_boxes` — one box per digit wheel, in left-to-right order, relative to the aligned crop

Start by capturing a test frame from each camera and measuring the coordinates in an image viewer (e.g. GIMP, Preview):

```bash
# List detected USB webcam device indices
meteocr list-webcams

# Grab a single frame for inspection (use predict with a dummy model, or just use OpenCV directly)
python - <<'EOF'
import cv2
cap = cv2.VideoCapture(0)   # adjust index
ret, frame = cap.read()
cv2.imwrite("test_frame_M1.png", frame)
cap.release()
EOF
```

Then update `configs/meters.yaml` with the measured coordinates.

---

## Bootstrap: collect labeled training data

For each meter, capture frames and label the full reading:

```bash
meteocr label-frame --meter M1 --image path/to/frame.png --reading 03506
meteocr label-frame --meter M2 --image path/to/frame.png --reading 01234
meteocr label-frame --meter M3 --image path/to/frame.png --reading 07890
```

This writes to `data/labels/frames.csv` and `data/labels/samples.csv`, and saves raw cell images and normalized images to `data/cells/` and `data/normalized/`.

Inspect the normalized images to verify that the digit shapes look clean and consistent. If they look wrong, adjust the meter geometry or normalization settings in `configs/`.

**Minimum to train a first model:** 10–20 labeled frames spread across different readings. More is better. The classifier is pooled, so a digit `3` from meter M1 helps classify `3` on M2 and M3.

---

## Train

```bash
meteocr train
```

Uses `data/labels/samples.csv` and saves the model to `models/digit_clf.joblib`.

To specify paths explicitly:

```bash
meteocr train --samples data/labels/samples.csv --model models/digit_clf.joblib
```

The model bundle contains the classifier, scaler, HOG config, and normalization config. The same bundle is used for all subsequent inference.

---

## Evaluate

```bash
meteocr evaluate
```

Splits samples by frame group (no leakage), runs the classifier on the validation set, and writes reports to `data/reports/`:

- `confusion_overall.csv`
- `accuracy_per_meter.csv`
- `accuracy_per_position.csv`
- `misclassified.csv`

---

## Predict a single frame

```bash
meteocr predict --meter M1 --image path/to/frame.png
```

Output:

```
Reading: 03506
Min confidence: 1.842
Mean confidence: 2.105
  pos 0: 0  (conf=2.311)
  pos 1: 3  (conf=1.842)
  pos 2: 5  (conf=2.044)
  pos 3: 0  (conf=2.198)
  pos 4: 6  (conf=2.132)
```

---

## Watch — continuous monitoring

### Live USB webcam

One process per meter. Match `--device` to the webcam index reported by `list-webcams`:

```bash
meteocr watch --meter M1 --device 0
meteocr watch --meter M2 --device 1
meteocr watch --meter M3 --device 2
```

The default capture interval is 60 seconds. Adjust with `--interval`:

```bash
meteocr watch --meter M1 --device 0 --interval 300
```

### Offline testing with test images

Put images in `data/test_images/M1/`, `data/test_images/M2/`, `data/test_images/M3/` (any common image format, sorted by filename).

```bash
meteocr watch --meter M1 --offline
```

Or point to a custom directory:

```bash
meteocr watch --meter M1 --offline --test-images path/to/images/
```

Add `--loop` to cycle through the images repeatedly:

```bash
meteocr watch --meter M1 --offline --loop --interval 0
```

### Watch output

Each processed frame produces a line:

```
[14:32:01] M1  raw=03506  stable=03506  [ACCEPTED]  conf_min=1.842
[14:32:01] M1  raw=03505  stable=03506  [HELD]      conf_min=0.231
```

Accepted readings are appended to `data/predictions/predictions.csv`.
Uncertain/rejected frames go to `data/labels/review_queue.csv` and optionally saved as images in `data/raw/uncertain/`.

### Confidence and plausibility tuning

```bash
meteocr watch --meter M1 --device 0 \
  --min-confidence 0.8 \
  --max-delta-hour 50
```

- `--min-confidence` — minimum per-digit decision margin to accept a reading (default: 0.5)
- `--max-delta-hour` — maximum plausible meter increment per hour; omit to disable (useful once you know your flow rate)

---

## Active learning loop

After the initial model is running, improve it without manual effort:

1. Uncertain frames accumulate in `data/labels/review_queue.csv`
2. Review them: check the image, note the correct reading
3. Re-label with `label-frame`
4. Retrain with `train`

Focus labeling effort on frames where the model was uncertain or wrong. The review queue already flags those.

---

## Configuration reference

### `configs/meters.yaml`

```yaml
meters:
  - meter_id: M1
    aligned_width: 500        # width of the canonical aligned crop (pixels)
    aligned_height: 120       # height of the canonical aligned crop (pixels)
    threshold_mode: otsu      # 'otsu' or 'adaptive'
    invert_binary: true       # invert after thresholding (true if digit is dark on light background)
    crop_source_box:          # rectangle in the raw camera frame
      x: 120
      y: 80
      w: 500
      h: 120
    max_translation_px: 4     # max drift correction (0 to disable)
    inner_pad_x: 2            # shrink each digit cell box inward (removes separator borders)
    inner_pad_y: 2
    digit_boxes:              # one entry per digit, left to right, relative to aligned crop
      - { x: 0,   y: 0, w: 92, h: 120 }
      - { x: 96,  y: 0, w: 92, h: 120 }
      - { x: 192, y: 0, w: 92, h: 120 }
      - { x: 288, y: 0, w: 92, h: 120 }
      - { x: 384, y: 0, w: 92, h: 120 }
```

For perspective-distorted meters, replace `crop_source_box` with `perspective_src_points` (four corner points of the meter face in the raw frame, top-left → top-right → bottom-right → bottom-left).

### `configs/defaults.yaml`

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
  min_component_area: 8      # minimum blob size in pixels (removes noise)
  bbox_margin_px: 2          # padding around the digit blob before centering
  output_width: 40
  output_height: 64
  foreground: white          # 'white' (digit bright) or 'black' (digit dark)

training:
  model_type: linear_svc     # 'linear_svc' or 'knn'
  test_size: 0.2
  group_by: frame            # split validation by 'frame', 'date', or 'meter'
  random_state: 42
  svc_c: 1.0
  use_standard_scaler: true
```

---

## Troubleshooting

**Normalization images look wrong (blobs on the wrong side, noise, empty)**

- Check `invert_binary`. If the digit wheel is dark text on a light background, set `invert_binary: false`. If it is light text on dark, set `true`.
- Check `inner_pad_x/y`. Too much padding clips the digit; too little includes the separator border.
- Increase `min_component_area` if noise specks are being included as digits.

**Classifier is confused on a specific digit position**

- Add more labeled frames that include readings with that digit in that position.
- Check the per-position report in `data/reports/accuracy_per_position.csv`.

**Many frames end up in the review queue**

- Lower `--min-confidence` (trades accuracy for acceptance rate).
- Add more training data, especially for the digits showing low confidence.
- Check normalization quality — if normalized images look inconsistent, fix the config first.

**Webcam index is wrong**

```bash
meteocr list-webcams
```

On Linux the indices may shift on reboot. Use udev rules to assign fixed device paths if needed.
