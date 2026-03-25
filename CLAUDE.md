# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**meterocr** is a classical computer vision pipeline for reading three indoor water meters from USB webcam images. It uses HOG features + LinearSVC (no deep learning), runs on Raspberry Pi, and includes an active-learning loop to improve over time.

## Common Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_predict.py -v

# Run with coverage
pytest tests/ -v --cov=src/meterocr

# CLI entry point
meterocr --help

# Capture a reference frame
meterocr capture-frame --meter M1 --output data/raw/ref_M1.png

# Label a frame (bootstraps training data)
meterocr label-frame --meter M1 --image data/raw/frame.png --reading 03506

# Train classifier
meterocr train

# Evaluate on validation split
meterocr evaluate

# Single-frame inference
meterocr predict --meter M1 --image path/to/frame.png

# Read all meters once and exit (designed for crontab)
meterocr read

# Read a single meter
meterocr read --meter M1

# Offline testing with test images in data/test_images/M1/
meterocr read --offline

# Offline loop mode (cycles through test images)
meterocr read --offline --loop
```

## Architecture

### Processing Pipeline

```
Raw frame (1920×1080 BGR)
  → align.py       Crop/warp to canonical 500×120 per meter
  → segment.py     Extract 5 fixed-position digit cells (geometry from meters.yaml)
  → normalize.py   Grayscale → Otsu/adaptive threshold → blob find → resize to 40×64
  → features.py    HOG feature vector (~324-dim float32)
  → predict.py     LinearSVC classifies each digit (0–9) + confidence margin
  → temporal.py    Monotonicity + confidence + plausible-delta filters
  → CSV output     data/predictions/predictions.csv
```

### Image Capture Resolution

All image capture throughout the app must use **1920×1080**. This applies to `WebcamCapture`, `capture_frame.py`, `capture_frame_stable.py`, and any new capture code. Lower resolutions make the meter digits harder to read reliably.

### Key Design Decisions

- **Single pooled classifier** trained on all 3 meters and all digit positions together — more data per class improves accuracy.
- **Geometry-based segmentation** uses fixed pixel coordinates from `configs/meters.yaml`. No dynamic detection; relies on stable camera mount + alignment.
- **Temporal monotonicity**: water meters only increase, so any reading decrease is rejected as a classification error.
- **Joblib model bundles** pack classifier + HOG config + normalization config together — inference automatically uses matching parameters.

### Module Responsibilities

| Module | Role |
|--------|------|
| `types.py` | All dataclasses (`MeterConfig`, `DigitCell`, `PredictionResult`, `MeterState`, etc.) |
| `cli.py` | Typer CLI entry points |
| `meter_config.py` | Load `configs/meters.yaml` and `configs/defaults.yaml` |
| `align.py` | Crop or perspective-warp frame to canonical size; optional translation drift correction |
| `segment.py` | Extract digit cells by fixed geometry |
| `normalize.py` | Threshold → remove noise → find blob → center/resize to 40×64 |
| `features.py` | HOG feature extraction (scikit-image) |
| `predict.py` | Full per-frame inference pipeline |
| `temporal.py` | Stateful filters: confidence, monotonicity, max-delta-per-hour |
| `train.py` | Load samples.csv → split by frame → fit LinearSVC → save joblib |
| `evaluate.py` | Confusion matrix, per-meter and per-position accuracy CSVs |
| `labeling.py` | Label a frame and write to frames.csv + samples.csv |
| `dataset.py` | CSV I/O for frames and digit samples |
| `capture.py` | `WebcamCapture` (USB, MJPG 1920×1080) and `TestImageCapture` (offline) |
| `model_io.py` | Save/load joblib model bundle |
| `review.py` | Track low-confidence frames in review_queue.csv |

### Configuration

- `configs/meters.yaml` — Per-meter: `video_device`, `crop_source_box`, `aligned_width/height`, `digit_boxes` (5 per meter), `threshold_mode`, `invert_binary`, `max_translation_px`
- `configs/defaults.yaml` — HOG params (40×64 image, 9 orientations), normalization params, training params (LinearSVC, 20% test split)

### Data Layout

```
data/
  raw/          Original captured frames
  aligned/      Cropped/warped canonical images
  cells/        Extracted digit cells
  normalized/   40×64 normalized digit images
  labels/
    frames.csv      Frame metadata and ground-truth readings
    samples.csv     Per-digit samples with labels (input to train)
    review_queue.csv  Uncertain frames flagged for review
  predictions/
    predictions.csv   Accepted stable readings from read
  reports/      Confusion matrices and accuracy CSVs
  test_images/  Offline test images (M1/, M2/, M3/ subdirs)
models/         Trained joblib bundles (gitignored)
```

## Workflow: Bootstrap to Deployment

1. Capture reference frames → measure pixel coordinates → update `configs/meters.yaml`
2. `label-frame` × 10–20 frames per meter across different readings
3. `train` → `evaluate` → check accuracy reports
4. Add `meterocr read` to crontab (e.g. `*/10 * * * * cd /path/to/meterocr && .venv/bin/meterocr read`)
5. Review `review_queue.csv` uncertain frames, re-label if needed, retrain to improve
