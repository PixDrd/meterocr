"""Temporal consistency layer: monotonicity, plausibility, and confidence checks."""

from datetime import datetime, timedelta

from meterocr.types import MeterState, PredictionResult, StableReadingResult


def reading_to_int(reading: str) -> int:
    """Convert a reading string to an integer for comparison.

    Args:
        reading: Digit string such as '03506'.

    Returns:
        Integer value.
    """
    return int(reading)


def update_meter_state(
    state: MeterState,
    prediction: PredictionResult,
    min_confidence_threshold: float,
    max_delta_per_hour: float | None = None,
    rolling_window_size: int = 5,
) -> StableReadingResult:
    """Apply temporal filters and update meter state.

    Rules applied in order:
    1. Confidence threshold: if min_confidence < threshold, hold.
    2. Monotonicity: if new reading < last stable, reject.
    3. Plausible delta: if jump exceeds max_delta_per_hour, hold.
    4. Accept reading and update stable state.

    Args:
        state: Current mutable meter state.
        prediction: New raw prediction from predict.py.
        min_confidence_threshold: Minimum acceptable per-digit confidence.
        max_delta_per_hour: Maximum plausible meter increment per hour, or None.
        rolling_window_size: How many recent predictions to keep.

    Returns:
        StableReadingResult indicating accept/reject/hold.
    """
    state.recent_predictions.append(prediction)
    prune_recent_predictions(state, rolling_window_size)

    new_reading = prediction.raw_reading
    stable = state.last_stable_reading

    if prediction.min_confidence < min_confidence_threshold:
        return StableReadingResult(
            frame_id=prediction.frame_id,
            meter_id=prediction.meter_id,
            timestamp=prediction.timestamp,
            accepted=False,
            raw_reading=new_reading,
            stable_reading=stable or new_reading,
            reason=f"confidence {prediction.min_confidence:.3f} below threshold {min_confidence_threshold:.3f}",
        )

    if not is_monotonic(stable, new_reading):
        return StableReadingResult(
            frame_id=prediction.frame_id,
            meter_id=prediction.meter_id,
            timestamp=prediction.timestamp,
            accepted=False,
            raw_reading=new_reading,
            stable_reading=stable or new_reading,
            reason=f"reading {new_reading} is less than last stable {stable}",
        )

    if not is_plausible_delta(
        stable,
        state.last_stable_timestamp,
        new_reading,
        prediction.timestamp,
        max_delta_per_hour,
    ):
        return StableReadingResult(
            frame_id=prediction.frame_id,
            meter_id=prediction.meter_id,
            timestamp=prediction.timestamp,
            accepted=False,
            raw_reading=new_reading,
            stable_reading=stable or new_reading,
            reason=f"implausible jump to {new_reading} from {stable}",
        )

    state.last_stable_reading = new_reading
    state.last_stable_timestamp = prediction.timestamp

    return StableReadingResult(
        frame_id=prediction.frame_id,
        meter_id=prediction.meter_id,
        timestamp=prediction.timestamp,
        accepted=True,
        raw_reading=new_reading,
        stable_reading=new_reading,
        reason="ok",
    )


def is_monotonic(last_reading: str | None, new_reading: str) -> bool:
    """Return True if new_reading >= last_reading (or no prior reading).

    Args:
        last_reading: Previous stable reading string, or None.
        new_reading: New candidate reading string.

    Returns:
        True if the reading is non-decreasing.
    """
    if last_reading is None:
        return True
    try:
        return reading_to_int(new_reading) >= reading_to_int(last_reading)
    except ValueError:
        return False


def is_plausible_delta(
    last_reading: str | None,
    last_timestamp: datetime | None,
    new_reading: str,
    new_timestamp: datetime,
    max_delta_per_hour: float | None,
) -> bool:
    """Return True if the increment is within the plausible rate.

    If max_delta_per_hour is None or no prior reading exists, always returns True.

    Args:
        last_reading: Previous stable reading string, or None.
        last_timestamp: Timestamp of the previous stable reading, or None.
        new_reading: New candidate reading string.
        new_timestamp: Timestamp of the new reading.
        max_delta_per_hour: Maximum allowed increment per hour, or None.

    Returns:
        True if the delta is plausible.
    """
    if max_delta_per_hour is None or last_reading is None or last_timestamp is None:
        return True
    try:
        delta_value = reading_to_int(new_reading) - reading_to_int(last_reading)
        elapsed = (new_timestamp - last_timestamp).total_seconds() / 3600.0
        if elapsed <= 0:
            return True
        rate = delta_value / elapsed
        return rate <= max_delta_per_hour
    except (ValueError, ZeroDivisionError):
        return True


def prune_recent_predictions(state: MeterState, max_items: int) -> None:
    """Trim the recent_predictions list to at most max_items entries.

    Args:
        state: Mutable meter state.
        max_items: Maximum number of predictions to retain.
    """
    if len(state.recent_predictions) > max_items:
        state.recent_predictions = state.recent_predictions[-max_items:]
