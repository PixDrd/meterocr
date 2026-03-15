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

from meterocr.capture import (
    CaptureError,
    MeterCaptureSet,
    TestImageCapture,
    WebcamCapture,
    build_capture_set,
    list_available_webcams,
)
from meterocr.labeling import label_frame
from meterocr.meter_config import get_meter_config, load_default_configs, load_meter_configs
from meterocr.model_io import load_model_bundle
from meterocr.predict import predict_meter_reading, predict_meter_reading_from_array
from meterocr.review import append_review_item
from meterocr.temporal import MeterState, update_meter_state
from meterocr.train import build_training_dataframe, split_train_validation, train_and_save_model
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

    _ensure_predictions_csv(predictions_csv)

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
