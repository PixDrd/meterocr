"""Command-line interface for the water meter OCR system."""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import cv2
import typer

import re

from meterocr.align import align_meter
from meterocr.segment import crop_digit_cell, extract_digit_cells
from meterocr.normalize import normalize_digit
from meterocr.capture import (
    CaptureError,
    MeterCaptureSet,
    TestImageCapture,
    WebcamCapture,
    build_capture_set,
    list_available_webcams,
)
from meterocr import dataset as _dataset
from meterocr.labeling import label_frame
from meterocr.types import DigitSample, FrameMeta
from meterocr.meter_config import get_meter_config, load_default_configs, load_meter_configs
from meterocr.model_io import load_model_bundle
from meterocr.predict import predict_meter_reading, predict_meter_reading_from_array
from meterocr.review import append_review_item
from meterocr.temporal import MeterState, update_meter_state
from meterocr.train import build_training_dataframe, split_train_validation, summarise_coverage, train_and_save_model
from meterocr.evaluate import evaluate_digit_classifier
from meterocr.utils import make_frame_id

app = typer.Typer(
    name="meterocr",
    help="Water meter OCR pipeline using HOG + scikit-learn.",
    add_completion=False,
)

# Default paths
_DEFAULT_CONFIGS = Path("configs/meters.yaml")
_DEFAULT_DEFAULTS = Path("configs/defaults.yaml")
_DEFAULT_FRAMES_CSV = Path("data/labels/frames.csv")
_DEFAULT_SAMPLES_CSV = Path("data/labels/samples.csv")
_DEFAULT_RAW_CELL_DIR = Path("data/cells")
_DEFAULT_NORMALIZED_DIR = Path("data/normalized")
_DEFAULT_MODEL = Path("models/digit_clf.joblib")
_DEFAULT_REPORTS_DIR = Path("data/reports")
_DEFAULT_REVIEW_CSV = Path("data/labels/review_queue.csv")
_DEFAULT_PREDICTIONS_CSV = Path("data/predictions/predictions.csv")
_DEFAULT_TEST_IMAGES_ROOT = Path("data/test_images")


@app.command("label-frame")
def cmd_label_frame(
    meter: Annotated[str, typer.Option("--meter", "-m", help="Meter ID (e.g. M1)")],
    image: Annotated[Path, typer.Option("--image", "-i", help="Path to frame image")],
    reading: Annotated[str, typer.Option("--reading", "-r", help="Full meter reading (digits only)")],
    configs: Annotated[Path, typer.Option(help="Path to meters.yaml")] = _DEFAULT_CONFIGS,
    defaults: Annotated[Path, typer.Option(help="Path to defaults.yaml")] = _DEFAULT_DEFAULTS,
    frames_csv: Annotated[Path, typer.Option(help="Path to frames.csv")] = _DEFAULT_FRAMES_CSV,
    samples_csv: Annotated[Path, typer.Option(help="Path to samples.csv")] = _DEFAULT_SAMPLES_CSV,
    raw_cell_dir: Annotated[Path, typer.Option(help="Raw cell output dir")] = _DEFAULT_RAW_CELL_DIR,
    normalized_dir: Annotated[Path, typer.Option(help="Normalized output dir")] = _DEFAULT_NORMALIZED_DIR,
) -> None:
    """Label a frame image with its full meter reading and extract digit samples."""
    meter_configs = load_meter_configs(configs)
    _, norm_cfg, _ = load_default_configs(defaults)
    meter_config = get_meter_config(meter_configs, meter)

    label_frame(
        image_path=image,
        meter_config=meter_config,
        normalization_cfg=norm_cfg,
        full_reading=reading,
        frames_csv=frames_csv,
        samples_csv=samples_csv,
        raw_cell_dir=raw_cell_dir,
        normalized_dir=normalized_dir,
    )
    typer.echo(f"Labeled frame {image} for meter {meter} with reading '{reading}'")


@app.command("train")
def cmd_train(
    samples_csv: Annotated[Path, typer.Option("--samples", help="Path to samples.csv")] = _DEFAULT_SAMPLES_CSV,
    model: Annotated[Path, typer.Option("--model", help="Output model path")] = _DEFAULT_MODEL,
    defaults: Annotated[Path, typer.Option(help="Path to defaults.yaml")] = _DEFAULT_DEFAULTS,
) -> None:
    """Train the digit classifier from labeled samples."""
    hog_cfg, norm_cfg, training_cfg = load_default_configs(defaults)
    df = build_training_dataframe(samples_csv)
    coverage = summarise_coverage(df)

    result = train_and_save_model(
        samples_csv=samples_csv,
        model_path=model,
        hog_cfg=hog_cfg,
        normalization_cfg=norm_cfg,
        training_cfg=training_cfg,
    )
    typer.echo(
        f"Trained {training_cfg.model_type} on {result['train_count']} samples "
        f"({result['feature_dim']} features). Model saved to {model}"
    )

    typer.echo("")
    typer.echo("=== Training data coverage ===")
    counts = coverage["per_digit"]
    typer.echo("Samples per digit (0-9):")
    typer.echo("  " + "  ".join(f"{d}:{counts[d]:>3}" for d in range(10)))

    if coverage["missing"]:
        typer.echo(f"  MISSING digits (no samples at all): {coverage['missing']}")
    if coverage["thin"]:
        typer.echo(f"  LOW    digits (< 5 samples):        {coverage['thin']}")
    if not coverage["missing"] and not coverage["thin"]:
        typer.echo("  All digits have sufficient samples.")

    if len(coverage["per_meter"]) > 1:
        typer.echo("Missing digits per meter:")
        for meter_id, missing in sorted(coverage["missing_per_meter"].items()):
            status = str(missing) if missing else "none"
            typer.echo(f"  {meter_id}: {status}")


@app.command("evaluate")
def cmd_evaluate(
    samples_csv: Annotated[Path, typer.Option("--samples", help="Path to samples.csv")] = _DEFAULT_SAMPLES_CSV,
    model: Annotated[Path, typer.Option("--model", help="Path to model bundle")] = _DEFAULT_MODEL,
    reports: Annotated[Path, typer.Option("--reports", help="Reports output directory")] = _DEFAULT_REPORTS_DIR,
    defaults: Annotated[Path, typer.Option(help="Path to defaults.yaml")] = _DEFAULT_DEFAULTS,
) -> None:
    """Evaluate the digit classifier on the validation split."""
    hog_cfg, norm_cfg, training_cfg = load_default_configs(defaults)
    bundle = load_model_bundle(model)

    df = build_training_dataframe(samples_csv)
    _, val_df = split_train_validation(df, training_cfg)

    if val_df.empty:
        typer.echo("Validation set is empty; add more labeled frames first.")
        raise typer.Exit(1)

    results = evaluate_digit_classifier(
        val_df=val_df,
        classifier=bundle["classifier"],
        scaler=bundle.get("scaler"),
        hog_cfg=bundle["hog_cfg"],
        reports_dir=reports,
    )
    typer.echo(f"Digit accuracy: {results['digit_accuracy']:.4f}")
    typer.echo(f"Per-meter: {results['per_meter']}")
    typer.echo(f"Per-position: {results['per_position']}")
    typer.echo(f"Reports saved to {reports}")


@app.command("predict")
def cmd_predict(
    meter: Annotated[str, typer.Option("--meter", "-m", help="Meter ID")],
    image: Annotated[Path, typer.Option("--image", "-i", help="Path to frame image")],
    model: Annotated[Path, typer.Option("--model", help="Path to model bundle")] = _DEFAULT_MODEL,
    configs: Annotated[Path, typer.Option(help="Path to meters.yaml")] = _DEFAULT_CONFIGS,
) -> None:
    """Predict the meter reading from a single frame image."""
    meter_configs = load_meter_configs(configs)
    meter_config = get_meter_config(meter_configs, meter)
    bundle = load_model_bundle(model)

    result = predict_meter_reading(image, meter_config, bundle)
    typer.echo(f"Reading: {result.raw_reading}")
    typer.echo(f"Min confidence: {result.min_confidence:.3f}")
    typer.echo(f"Mean confidence: {result.mean_confidence:.3f}")
    for dp in result.digits:
        typer.echo(f"  pos {dp.position}: {dp.digit}  (conf={dp.confidence:.3f})")


@app.command("watch")
def cmd_watch(
    meter: Annotated[str, typer.Option("--meter", "-m", help="Meter ID")],
    model: Annotated[Path, typer.Option("--model", help="Path to model bundle")] = _DEFAULT_MODEL,
    configs: Annotated[Path, typer.Option(help="Path to meters.yaml")] = _DEFAULT_CONFIGS,
    defaults: Annotated[Path, typer.Option(help="Path to defaults.yaml")] = _DEFAULT_DEFAULTS,
    device: Annotated[Optional[str], typer.Option("--device", help="Override webcam device (e.g. /dev/video0); uses meters.yaml value by default")] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Use test images instead of webcam")] = False,
    test_images: Annotated[Optional[Path], typer.Option("--test-images", help="Directory with test images for this meter")] = None,
    interval: Annotated[float, typer.Option("--interval", help="Seconds between captures")] = 60.0,
    min_confidence: Annotated[float, typer.Option("--min-confidence", help="Confidence threshold")] = 0.5,
    max_delta_hour: Annotated[Optional[float], typer.Option("--max-delta-hour", help="Max plausible increment per hour")] = None,
    predictions_csv: Annotated[Path, typer.Option("--predictions-csv", help="Output predictions CSV")] = _DEFAULT_PREDICTIONS_CSV,
    review_csv: Annotated[Path, typer.Option("--review-csv", help="Review queue CSV")] = _DEFAULT_REVIEW_CSV,
    save_uncertain: Annotated[bool, typer.Option("--save-uncertain", help="Save uncertain frame images")] = True,
    uncertain_dir: Annotated[Path, typer.Option("--uncertain-dir", help="Directory for uncertain frames")] = Path("data/raw/uncertain"),
    unknown_digits_dir: Annotated[Path, typer.Option("--unknown-digits-dir", help="Directory to save unreadable digit crops")] = Path("data/unknown_digits"),
    loop: Annotated[bool, typer.Option("--loop", help="Loop test images indefinitely")] = False,
) -> None:
    """Watch a meter and emit periodic stable readings.

    The webcam device is read from meters.yaml (video_device field).
    Use --device to override it on the command line.
    Use --offline with --test-images for offline testing.
    """
    meter_configs = load_meter_configs(configs)
    _, norm_cfg, _ = load_default_configs(defaults)
    meter_config = get_meter_config(meter_configs, meter)
    bundle = load_model_bundle(model)

    # Resolve capture source: CLI flag > meters.yaml > offline fallback
    if offline:
        img_dir = test_images or (_DEFAULT_TEST_IMAGES_ROOT / meter)
        capture = TestImageCapture(img_dir, loop=loop)
        typer.echo(f"Offline mode: reading test images from {img_dir}")
    else:
        resolved_device: str | int | None = device or meter_config.video_device
        if resolved_device is None:
            typer.echo(
                f"Error: no video_device configured for meter {meter} in meters.yaml "
                "and --device was not provided. Use --offline for test images.",
                err=True,
            )
            raise typer.Exit(1)
        capture = WebcamCapture(resolved_device)
        typer.echo(f"Live mode: webcam device '{resolved_device}'")

    state = MeterState(meter_id=meter)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    uncertain_dir.mkdir(parents=True, exist_ok=True)
    unknown_digits_dir.mkdir(parents=True, exist_ok=True)

    _ensure_predictions_csv(predictions_csv)

    unknown_seq = _next_unknown_seq(unknown_digits_dir, meter)

    typer.echo(f"Watching meter {meter}. Press Ctrl+C to stop.")
    frame_count = 0

    try:
        with capture:
            while True:
                try:
                    frame_bgr = capture.grab_frame()
                except CaptureError as e:
                    typer.echo(f"Capture stopped: {e}")
                    break

                frame_count += 1
                ts = datetime.now()
                frame_id = make_frame_id(meter, ts)

                prediction = predict_meter_reading_from_array(
                    frame_bgr, frame_id, ts, meter_config, bundle
                )
                stable = update_meter_state(
                    state,
                    prediction,
                    min_confidence_threshold=min_confidence,
                    max_delta_per_hour=max_delta_hour,
                )

                status = "ACCEPTED" if stable.accepted else "HELD"
                typer.echo(
                    f"[{ts.strftime('%H:%M:%S')}] {meter}  raw={prediction.raw_reading}"
                    f"  stable={stable.stable_reading}  [{status}]"
                    f"  conf_min={prediction.min_confidence:.3f}"
                )

                _append_prediction_row(predictions_csv, prediction, stable)

                if not stable.accepted and save_uncertain:
                    uncertain_path = uncertain_dir / f"{frame_id}.png"
                    cv2.imwrite(str(uncertain_path), frame_bgr)
                    append_review_item(review_csv, prediction, stable, uncertain_path)

                # Save individual digit crops for any digit below the confidence threshold
                low_conf_digits = [dp for dp in prediction.digits if dp.confidence < min_confidence]
                if low_conf_digits:
                    aligned = align_meter(frame_bgr, meter_config)
                    frame_meta = FrameMeta(
                        frame_id=frame_id,
                        meter_id=meter,
                        timestamp=ts,
                        source_path=Path(frame_id),
                    )
                    cells = extract_digit_cells(aligned, meter_config, frame_meta)
                    for dp in low_conf_digits:
                        cell_img = cells[dp.position].image_bgr
                        fname = f"{meter}_{unknown_seq:04d}_pos{dp.position}_unknown.png"
                        cv2.imwrite(str(unknown_digits_dir / fname), cell_img)
                        unknown_seq += 1

                if not offline or loop:
                    time.sleep(interval)
                else:
                    # Offline non-looping: no sleep, consume all images quickly
                    pass

    except KeyboardInterrupt:
        typer.echo(f"\nStopped after {frame_count} frames.")


@app.command("capture-frame")
def cmd_capture_frame(
    meter: Annotated[str, typer.Option("--meter", "-m", help="Meter ID")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output image path")],
    configs: Annotated[Path, typer.Option(help="Path to meters.yaml")] = _DEFAULT_CONFIGS,
    device: Annotated[Optional[str], typer.Option("--device", help="Override webcam device")] = None,
) -> None:
    """Capture a single frame from a meter's webcam and save it to disk.

    Useful for grabbing a reference image to measure crop coordinates.
    """
    meter_configs = load_meter_configs(configs)
    meter_config = get_meter_config(meter_configs, meter)

    resolved_device: str | int | None = device or meter_config.video_device
    if resolved_device is None:
        typer.echo(
            f"Error: no video_device configured for meter {meter} in meters.yaml "
            "and --device was not provided.",
            err=True,
        )
        raise typer.Exit(1)

    with WebcamCapture(resolved_device) as cap:
        frame = cap.grab_frame()

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    typer.echo(f"Saved frame to {output}  ({frame.shape[1]}x{frame.shape[0]})")


@app.command("crop-test")
def cmd_crop_test(
    meter: Annotated[str, typer.Option("--meter", "-m", help="Meter ID (e.g. M1)")],
    image: Annotated[Optional[Path], typer.Option("--image", "-i", help="Input frame; omit to capture from webcam")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output path (default: data/aligned/<meter>_crop_test.png)")] = None,
    configs: Annotated[Path, typer.Option(help="Path to meters.yaml")] = _DEFAULT_CONFIGS,
    device: Annotated[Optional[str], typer.Option("--device", help="Override webcam device")] = None,
) -> None:
    """Crop and perspective-correct a frame and save the result for inspection.

    Pass an existing image with --image, or omit it to capture a fresh frame
    from the meter's configured webcam.
    """
    meter_configs = load_meter_configs(configs)
    meter_config = get_meter_config(meter_configs, meter)

    if image is not None:
        frame = cv2.imread(str(image))
        if frame is None:
            typer.echo(f"Error: could not read image {image}", err=True)
            raise typer.Exit(1)
    else:
        resolved_device: str | int | None = device or meter_config.video_device
        if resolved_device is None:
            typer.echo(
                f"Error: no video_device configured for meter {meter} in meters.yaml "
                "and --device was not provided.",
                err=True,
            )
            raise typer.Exit(1)
        with WebcamCapture(resolved_device) as cap:
            frame = cap.grab_frame()

    aligned = align_meter(frame, meter_config)

    out_path = output or Path("data/aligned") / f"{meter}_crop_test.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), aligned)
    typer.echo(f"Saved aligned crop to {out_path}  ({aligned.shape[1]}x{aligned.shape[0]})")

    stem = out_path.stem
    for i, box in enumerate(meter_config.digit_boxes, start=1):
        cell = crop_digit_cell(aligned, box.x, box.y, box.w, box.h)
        cell_path = out_path.with_name(f"{stem}_{i}{out_path.suffix}")
        cv2.imwrite(str(cell_path), cell)
        typer.echo(f"  digit {i}: {cell_path}  ({cell.shape[1]}x{cell.shape[0]})")


@app.command("list-webcams")
def cmd_list_webcams(
    max_index: Annotated[int, typer.Option("--max-index", help="Max device index to probe")] = 10,
) -> None:
    """List available USB webcam device indices."""
    indices = list_available_webcams(max_index)
    if not indices:
        typer.echo("No webcams found.")
    else:
        typer.echo(f"Available webcam indices: {indices}")


@app.command("import-unknown")
def cmd_import_unknown(
    unknown_dir: Annotated[Path, typer.Argument(help="Directory containing renamed digit crops")],
    configs: Annotated[Path, typer.Option(help="Path to meters.yaml")] = _DEFAULT_CONFIGS,
    defaults: Annotated[Path, typer.Option(help="Path to defaults.yaml")] = _DEFAULT_DEFAULTS,
    samples_csv: Annotated[Path, typer.Option("--samples", help="Path to samples.csv")] = _DEFAULT_SAMPLES_CSV,
    normalized_dir: Annotated[Path, typer.Option(help="Directory to save normalized images")] = _DEFAULT_NORMALIZED_DIR,
) -> None:
    """Import renamed unknown digit crops into the training dataset.

    Rename files from M1_0001_pos2_unknown.png to M1_0001_pos2_7.png,
    then run this command to normalize and add them to samples.csv.
    """
    meter_configs = load_meter_configs(configs)
    _, norm_cfg, _ = load_default_configs(defaults)

    pattern = re.compile(r"^([^_]+)_(\d+)_pos(\d+)_([0-9])\.png$", re.IGNORECASE)
    files = sorted(unknown_dir.glob("*.png"))
    imported = 0
    skipped = 0

    for f in files:
        m = pattern.match(f.name)
        if not m:
            skipped += 1
            continue

        meter_id, seq, position, digit_char = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))

        try:
            meter_config = get_meter_config(meter_configs, meter_id)
        except Exception:
            typer.echo(f"  SKIP {f.name}: unknown meter '{meter_id}'")
            skipped += 1
            continue

        cell_bgr = cv2.imread(str(f))
        if cell_bgr is None:
            typer.echo(f"  SKIP {f.name}: could not read image")
            skipped += 1
            continue

        norm_result = normalize_digit(cell_bgr, meter_config.threshold_mode, meter_config.invert_binary, norm_cfg)

        normalized_dir.mkdir(parents=True, exist_ok=True)
        norm_stem = f"{meter_id}_{seq:04d}_pos{position}_imported"
        norm_path = normalized_dir / f"{norm_stem}_norm.png"
        cv2.imwrite(str(norm_path), norm_result.normalized)

        sample = DigitSample(
            frame_id=f"imported_{meter_id}_{seq:04d}",
            meter_id=meter_id,
            timestamp=datetime.now(),
            position=position,
            digit_label=digit_char,
            full_reading="",
            raw_cell_path=f,
            normalized_path=norm_path,
            quality="ok" if norm_result.success else "uncertain",
        )
        _dataset.append_digit_samples(samples_csv, [sample])
        typer.echo(f"  {f.name}  →  digit={digit_char}  pos={position}  quality={sample.quality}")
        imported += 1

    typer.echo(f"\nImported {imported} samples, skipped {skipped} files.")


def _next_unknown_seq(directory: Path, meter_id: str) -> int:
    """Return the next sequence number for unknown digit crops in directory."""
    pattern = re.compile(rf"^{re.escape(meter_id)}_(\d+)_pos\d+_(unknown|\d)\.png$", re.IGNORECASE)
    max_seq = 0
    for f in directory.glob(f"{meter_id}_*.png"):
        m = pattern.match(f.name)
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    return max_seq + 1


def _ensure_predictions_csv(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame_id", "timestamp", "meter_id", "raw_reading",
                "stable_reading", "accepted", "reason",
                "min_confidence", "mean_confidence",
            ])
            writer.writeheader()


def _append_prediction_row(path: Path, prediction, stable) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "timestamp", "meter_id", "raw_reading",
            "stable_reading", "accepted", "reason",
            "min_confidence", "mean_confidence",
        ])
        writer.writerow({
            "frame_id": prediction.frame_id,
            "timestamp": prediction.timestamp.isoformat(),
            "meter_id": prediction.meter_id,
            "raw_reading": prediction.raw_reading,
            "stable_reading": stable.stable_reading,
            "accepted": str(stable.accepted),
            "reason": stable.reason,
            "min_confidence": f"{prediction.min_confidence:.4f}",
            "mean_confidence": f"{prediction.mean_confidence:.4f}",
        })


if __name__ == "__main__":
    app()
