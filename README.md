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

Install virtualenv:

```python3 -m venv .venv
```

Activate the venv:

```source .venv/bin/activate
```

Install dependencies:

```pip install -e ".[dev]"
```

Check that anything works:

```meterocr test
```

Should reply with "hello".

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

- `video_device` — the webcam device path for this meter (e.g. `/dev/video0`)
- `crop_source_box` — the rectangle in the raw camera frame that contains the meter display (`x, y, w, h` in pixels); use this when the camera is roughly perpendicular to the meter face
- `aligned_width` / `aligned_height` — the canonical size you want to work with
- `perspective_src_points` — use instead of `crop_source_box` when the camera is angled; specify the four corners of the meter face in the raw frame (see Configuration reference)
- `digit_boxes` — one box per digit wheel, in left-to-right order, relative to the aligned crop

Start by identifying which device belongs to which meter:

```bash
meterocr list-webcams
```

Then capture a reference frame from each camera to measure coordinates in an image viewer (e.g. GIMP, Preview):

```bash
meterocr capture-frame --meter M1 --output data/raw/ref_M1.png
meterocr capture-frame --meter M2 --output data/raw/ref_M2.png
meterocr capture-frame --meter M3 --output data/raw/ref_M3.png
```

Pass `--device` to override the webcam device without editing `meters.yaml`:

```bash
meterocr capture-frame --meter M1 --output data/raw/ref_M1.png --device /dev/video4
```

Then update `configs/meters.yaml` with the measured coordinates.

### Verify crop coordinates with `crop-test`

After updating coordinates, verify the alignment without running the full pipeline:

```bash
# Capture a fresh frame and show the aligned crop + digit cells
meterocr crop-test --meter M1

# Use an existing image instead of capturing
meterocr crop-test --meter M1 --image data/raw/ref_M1.png

# Save to a custom path
meterocr crop-test --meter M1 --image data/raw/ref_M1.png --output data/aligned/check_M1.png
```

This saves the aligned crop and one image per digit cell (e.g. `check_M1_1.png` … `check_M1_5.png`). Inspect them to confirm the digit cells are correctly positioned before labeling.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--meter` / `-m` | required | Meter ID |
| `--image` / `-i` | — | Input frame; omit to capture from webcam |
| `--output` / `-o` | `data/aligned/<meter>_crop_test.png` | Output path for aligned crop |
| `--device` | from meters.yaml | Override webcam device |
| `--configs` | `configs/meters.yaml` | Path to meters.yaml |

---

## Bootstrap: collect labeled training data

For each meter, capture frames and label the full reading:

```bash
meterocr label-frame --meter M1 --image path/to/frame.png --reading 03506
meterocr label-frame --meter M2 --image path/to/frame.png --reading 01234
meterocr label-frame --meter M3 --image path/to/frame.png --reading 07890
```

This writes to `data/labels/frames.csv` and `data/labels/samples.csv`, and saves raw cell images and normalized images to `data/cells/` and `data/normalized/`.

Inspect the normalized images to verify that the digit shapes look clean and consistent. If they look wrong, adjust the meter geometry or normalization settings in `configs/`.

**Minimum to train a first model:** 10–20 labeled frames spread across different readings. More is better. The classifier is pooled, so a digit `3` from meter M1 helps classify `3` on M2 and M3.

---

## Train

```bash
meterocr train
```

Uses `data/labels/samples.csv` and saves the model to `models/digit_clf.joblib`.

To specify paths explicitly:

```bash
meterocr train --samples data/labels/samples.csv --model models/digit_clf.joblib
```

The model bundle contains the classifier, scaler, HOG config, and normalization config. The same bundle is used for all subsequent inference.

---

## Evaluate

```bash
meterocr evaluate
```

Splits samples by frame group (no leakage), runs the classifier on the validation set, and writes reports to `data/reports/`:

- `confusion_overall.csv`
- `accuracy_per_meter.csv`
- `accuracy_per_position.csv`
- `misclassified.csv`

---

## Predict a single frame

```bash
meterocr predict --meter M1 --image path/to/frame.png
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

## Read — one-shot meter reading

`meterocr read` reads all meters once and exits. Use crontab to control the schedule.

Temporal state (monotonicity, last stable reading) is bootstrapped from `predictions.csv` at startup, so each invocation continues where the last left off.

### Crontab setup

```cron
# Read all meters every 10 minutes
*/10 * * * * cd /home/pi/meterocr && .venv/bin/meterocr read
```

### Live USB webcam

The webcam device is read from `video_device` in `meters.yaml`, so normally no extra flags are needed:

```bash
meterocr read
```

Read a single meter:

```bash
meterocr read --meter M1
```

Override the device at runtime if needed (single-meter only):

```bash
meterocr read --meter M1 --device /dev/video0
```

### Offline testing with test images

Put images in `data/test_images/M1/`, `data/test_images/M2/`, `data/test_images/M3/` (any common image format, sorted by filename).

```bash
meterocr read --offline
```

Or point to a custom directory (single-meter):

```bash
meterocr read --meter M1 --offline --test-images path/to/images/
```

Add `--loop` to cycle through the images repeatedly:

```bash
meterocr read --offline --loop
```

### Read output

Each processed frame produces a line:

```
[14:32:01] M1  raw=03506  stable=03506  [ACCEPTED]  conf_min=1.842
[14:32:01] M1  raw=03505  stable=03506  [HELD]      conf_min=0.231
```

Accepted readings are appended to `data/predictions/predictions.csv`.
Uncertain/rejected frames go to `data/labels/review_queue.csv` and optionally saved as images in `data/raw/uncertain/`.

### Lighting control

If you control a light around each capture (e.g. via a smart plug or GPIO), use the pre/post command hooks:

```bash
meterocr read \
  --pre-read-cmd "python turn_light_on.py" \
  --post-read-cmd "python turn_light_off.py" \
  --warmup-secs 2.0
```

`--warmup-secs` (default: 2.0) is the delay between the pre-command finishing and the actual capture, giving the light time to stabilise.

### Full `read` option reference

| Flag | Default | Description |
|------|---------|-------------|
| `--meter` / `-m` | all meters | Meter ID to read; omit to read all sequentially |
| `--model` | `models/digit_clf.joblib` | Path to model bundle |
| `--configs` | `configs/meters.yaml` | Path to meters.yaml |
| `--defaults` | `configs/defaults.yaml` | Path to defaults.yaml |
| `--device` | from meters.yaml | Override webcam device (single-meter only) |
| `--offline` | false | Use test images instead of webcam |
| `--test-images` | `data/test_images/<meter>/` | Test image directory (single-meter offline only) |
| `--loop` | false | Loop test images indefinitely (offline only) |
| `--min-confidence` | `0.5` | Minimum per-digit decision margin to accept a reading |
| `--max-delta-hour` | disabled | Maximum plausible meter increment per hour |
| `--predictions-csv` | `data/predictions/predictions.csv` | Output accepted readings CSV (also used to bootstrap temporal state) |
| `--review-csv` | `data/labels/review_queue.csv` | Review queue CSV for held frames |
| `--save-uncertain` | true | Save image of uncertain/rejected frames |
| `--uncertain-dir` | `data/raw/uncertain/` | Directory for uncertain frame images |
| `--unknown-digits-dir` | `data/unknown_digits/` | Directory for low-confidence digit crops |
| `--latest-dir` | `data/latest/` | Updated each run with the latest aligned crop, raw crop, and full frame per meter |
| `--pre-read-cmd` | — | Shell command run before capture |
| `--post-read-cmd` | — | Shell command run after capture |
| `--warmup-secs` | `2.0` | Seconds to wait after `--pre-read-cmd` before capturing |

---

## Active learning loop

After the initial model is running, improve it without manual effort:

1. Uncertain frames accumulate in `data/labels/review_queue.csv`
2. Review them: check the image, note the correct reading
3. Re-label with `label-frame`
4. Retrain with `train`

Focus labeling effort on frames where the model was uncertain or wrong. The review queue already flags those.

### Importing low-confidence digit crops

`read` saves individual digit crops it could not classify confidently to `data/unknown_digits/` with names like `M1_0001_pos2_unknown.png`. To turn these into training samples:

1. Open the directory and rename each file, replacing `unknown` with the correct digit:
   `M1_0001_pos2_unknown.png` → `M1_0001_pos2_7.png`
2. Run `import-unknown` to normalise and append them to `samples.csv`:

```bash
meterocr import-unknown data/unknown_digits/
```

3. Retrain:

```bash
meterocr train
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--samples` | `data/labels/samples.csv` | Output samples CSV |
| `--normalized-dir` | `data/normalized/` | Where to save normalised images |
| `--configs` | `configs/meters.yaml` | Path to meters.yaml |
| `--defaults` | `configs/defaults.yaml` | Path to defaults.yaml |

---

## Configuration reference

### `configs/meters.yaml`

```yaml
meters:
  - meter_id: M1
    video_device: /dev/video0  # full device path for the USB webcam for this meter
    focus: 95                  # manual focus value 0–255; omit to leave autofocus enabled
    focus_settle_s: 8.0        # seconds to wait after setting focus before capturing (default 8.0)
    aligned_width: 500         # width of the canonical aligned crop (pixels)
    aligned_height: 120        # height of the canonical aligned crop (pixels)
    threshold_mode: otsu       # 'otsu' or 'adaptive'
    invert_binary: true        # true if digit is dark on a light background
    crop_source_box:           # rectangle in the raw camera frame
      x: 120
      y: 80
      w: 500
      h: 120
    max_translation_px: 4      # max drift correction in pixels (0 to disable)
    alignment_reference_path: data/raw/ref_M1.png  # reference image for drift correction; required when max_translation_px > 0
    inner_pad_x: 2             # shrink each digit cell box inward (removes separator borders)
    inner_pad_y: 2
    digit_boxes:               # one entry per digit, left to right, relative to aligned crop
      - { x: 0,   y: 0, w: 92, h: 120 }
      - { x: 96,  y: 0, w: 92, h: 120 }
      - { x: 192, y: 0, w: 92, h: 120 }
      - { x: 288, y: 0, w: 92, h: 120 }
      - { x: 384, y: 0, w: 92, h: 120 }
```

`video_device` should be the full device path (e.g. `/dev/video0`, `/dev/video2`). This keeps the meter-to-camera assignment stable across reboots when combined with udev rules. You can override it at runtime with `--device`.

`focus` sets manual focus. The V4L2 focus scale is typically 0–255; the optimal value depends on the lens and distance. Use `capture_frame_stable.py --focus-test 0 255` to sweep focus values and find the sharpest setting. When `focus` is set, `focus_settle_s` controls how long to wait after commanding the focus position before grabbing a frame — the motor needs time to reach its target.

If the camera is mounted at an angle to the meter face, use `perspective_src_points` instead of `crop_source_box`. Provide the four corners of the meter display in the raw frame — the code applies a perspective warp to straighten them into the canonical rectangle.

Point order: **top-left → top-right → bottom-right → bottom-left**, measured in the raw 1920×1080 frame.

```yaml
    perspective_src_points:   # mutually exclusive with crop_source_box
      - [142, 95]             # top-left
      - [638, 82]             # top-right
      - [644, 205]            # bottom-right
      - [136, 218]            # bottom-left
```

Measure the coordinates by opening a reference frame (captured with `capture-frame`) in an image viewer that shows pixel coordinates (e.g. GIMP → pointer tool, macOS Preview → Tools → Inspector). Click each corner of the meter face and note the `x, y` values. The four points do not need to form a perfect rectangle — any quadrilateral works.

`crop_source_box` and `perspective_src_points` are mutually exclusive; set only one per meter.

### `configs/defaults.yaml`

```yaml
hog:
  image_width: 40
  image_height: 64
  orientations: 9
  pixels_per_cell: [4, 4]
  cells_per_block: [2, 2]
  block_norm: L2-Hys         # 'L2-Hys', 'L1', 'L1-sqrt', or 'L2'
  transform_sqrt: false

normalization:
  blur_kernel: 3             # Gaussian blur kernel size (odd integer; 0 to disable)
  min_component_area: 8      # minimum blob size in pixels (removes noise)
  bbox_margin_px: 2          # padding around the digit blob before centering
  output_width: 40
  output_height: 64
  foreground: white          # 'white' (digit bright on dark) or 'black' (digit dark on bright)

training:
  model_type: linear_svc     # 'linear_svc' or 'knn'
  test_size: 0.2             # fraction of frames held out for validation
  group_by: frame            # validation split grouping: 'frame', 'date', or 'meter'
  random_state: 42
  svc_c: 1.0                 # LinearSVC regularisation strength (higher = less regularisation)
  use_standard_scaler: true  # normalise HOG features before classification (recommended)
  knn_neighbors: 3           # number of neighbours (only used when model_type: knn)
  knn_weights: distance      # 'uniform' or 'distance' (only used when model_type: knn)
```

---

## Troubleshooting

**Normalization images look wrong (blobs on the wrong side, noise, empty)**

- Check `invert_binary`. After Otsu thresholding, bright pixels become foreground. If the digit wheel has dark text on a light background, set `invert_binary: true` to flip the result so the digit is the bright foreground. If it has light text on a dark background, set `invert_binary: false`.
- Check `inner_pad_x/y`. Too much padding clips the digit; too little includes the separator border.
- Increase `min_component_area` if noise specks are being included as digits.

**Classifier is confused on a specific digit position**

- Add more labeled frames that include readings with that digit in that position.
- Check the per-position report in `data/reports/accuracy_per_position.csv`.

**Many frames end up in the review queue**

- Lower `--min-confidence` (trades accuracy for acceptance rate).
- Add more training data, especially for the digits showing low confidence.
- Check normalization quality — if normalized images look inconsistent, fix the config first.

**Wrong camera is being used for a meter**

```bash
meterocr list-webcams
meterocr capture-frame --meter M1 --output /tmp/test.png
```

Check the saved image to confirm it shows the right meter. Update `video_device` in `meters.yaml` if not. On Linux, camera indices can shift on reboot — use udev rules to assign stable device paths by USB port or serial number.
