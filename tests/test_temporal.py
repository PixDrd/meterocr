"""Tests for temporal.py."""

from datetime import datetime, timedelta

import pytest

from meterocr.temporal import (
    is_monotonic,
    is_plausible_delta,
    reading_to_int,
    update_meter_state,
)
from meterocr.types import DigitPrediction, MeterState, PredictionResult


def _make_prediction(reading: str, meter_id: str = "M1", confidence: float = 1.0) -> PredictionResult:
    digits = [
        DigitPrediction(position=i, digit=int(c), confidence=confidence, margin=confidence)
        for i, c in enumerate(reading)
    ]
    return PredictionResult(
        frame_id="f",
        meter_id=meter_id,
        timestamp=datetime.now(),
        raw_reading=reading,
        digits=digits,
        min_confidence=confidence,
        mean_confidence=confidence,
    )


def test_backwards_reading_is_rejected():
    state = MeterState(meter_id="M1", last_stable_reading="05000")
    pred = _make_prediction("04999")
    result = update_meter_state(state, pred, min_confidence_threshold=0.0)
    assert result.accepted is False
    assert "less than" in result.reason


def test_low_confidence_is_held():
    state = MeterState(meter_id="M1", last_stable_reading="05000")
    pred = _make_prediction("05001", confidence=0.1)
    result = update_meter_state(state, pred, min_confidence_threshold=0.5)
    assert result.accepted is False
    assert "confidence" in result.reason


def test_monotonic_plausible_is_accepted():
    state = MeterState(meter_id="M1", last_stable_reading="05000")
    pred = _make_prediction("05001", confidence=1.0)
    result = update_meter_state(state, pred, min_confidence_threshold=0.5)
    assert result.accepted is True
    assert result.stable_reading == "05001"


def test_first_reading_accepted_without_prior():
    state = MeterState(meter_id="M1")
    pred = _make_prediction("00123", confidence=0.9)
    result = update_meter_state(state, pred, min_confidence_threshold=0.5)
    assert result.accepted is True


def test_is_monotonic_no_prior():
    assert is_monotonic(None, "12345") is True


def test_is_monotonic_equal():
    assert is_monotonic("12345", "12345") is True


def test_is_monotonic_backwards():
    assert is_monotonic("12345", "12344") is False


def test_is_plausible_delta_no_prior():
    assert is_plausible_delta(None, None, "100", datetime.now(), 10.0) is True


def test_is_plausible_delta_within_limit():
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    t1 = datetime(2024, 1, 1, 1, 0, 0)  # 1 hour later
    assert is_plausible_delta("1000", t0, "1005", t1, max_delta_per_hour=10.0) is True


def test_is_plausible_delta_exceeds_limit():
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    t1 = datetime(2024, 1, 1, 1, 0, 0)
    assert is_plausible_delta("1000", t0, "2000", t1, max_delta_per_hour=10.0) is False


def test_state_updates_stable_reading_after_accept():
    state = MeterState(meter_id="M1", last_stable_reading="00100")
    pred = _make_prediction("00200", confidence=1.0)
    update_meter_state(state, pred, min_confidence_threshold=0.5)
    assert state.last_stable_reading == "00200"


def test_state_does_not_update_after_reject():
    state = MeterState(meter_id="M1", last_stable_reading="00100")
    pred = _make_prediction("00099", confidence=1.0)
    update_meter_state(state, pred, min_confidence_threshold=0.5)
    assert state.last_stable_reading == "00100"
