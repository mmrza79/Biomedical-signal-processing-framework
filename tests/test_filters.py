"""Unit tests using small synthetic signals only for mathematical verification."""

import numpy as np
import pytest

from src.preprocessing.filters import (
    bandpass_filter,
    normalize_signal,
    notch_filter,
    rectify_signal,
    remove_dc,
)
from src.preprocessing.segmentation import segment_signal


def _tone_amplitude(signal: np.ndarray, frequency_hz: float, sampling_frequency_hz: float) -> float:
    """Estimate a test tone's amplitude using its complex projection."""

    time = np.arange(signal.size) / sampling_frequency_hz
    projection = np.sum(signal * np.exp(-2j * np.pi * frequency_hz * time))
    return float(2 * np.abs(projection) / signal.size)


def test_remove_dc_centers_each_channel() -> None:
    signal = np.array([[2.0, 3.0, 4.0], [10.0, 12.0, 14.0]])

    centered = remove_dc(signal, axis=1)

    np.testing.assert_allclose(np.mean(centered, axis=1), 0.0, atol=1e-12)


def test_bandpass_preserves_passband_and_attenuates_low_frequency() -> None:
    sampling_frequency_hz = 1_000.0
    time = np.arange(0, 2, 1 / sampling_frequency_hz)
    signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 100 * time)

    filtered = bandpass_filter(signal, sampling_frequency_hz, 50.0, 150.0)

    assert _tone_amplitude(filtered, 100.0, sampling_frequency_hz) > 0.8
    assert _tone_amplitude(filtered, 10.0, sampling_frequency_hz) < 0.05


def test_bandpass_rejects_cutoff_at_or_above_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass_filter(np.arange(100, dtype=float), 1_000.0, 20.0, 500.0)


def test_notch_attenuates_configured_interference() -> None:
    sampling_frequency_hz = 1_000.0
    time = np.arange(0, 2, 1 / sampling_frequency_hz)
    signal = np.sin(2 * np.pi * 50 * time) + np.sin(2 * np.pi * 120 * time)

    filtered = notch_filter(signal, sampling_frequency_hz, 50.0, quality_factor=30.0)

    assert _tone_amplitude(filtered, 50.0, sampling_frequency_hz) < 0.1
    assert _tone_amplitude(filtered, 120.0, sampling_frequency_hz) > 0.8


def test_rectification_and_normalization() -> None:
    signal = np.array([-2.0, 0.0, 2.0])

    np.testing.assert_array_equal(rectify_signal(signal), [2.0, 0.0, 2.0])
    normalized = normalize_signal(signal, method="zscore")
    assert np.mean(normalized) == pytest.approx(0.0)
    assert np.std(normalized) == pytest.approx(1.0)


def test_normalize_constant_signal_is_finite() -> None:
    normalized = normalize_signal(np.ones(5), method="zscore")

    np.testing.assert_array_equal(normalized, np.zeros(5))


def test_segment_signal_has_expected_overlap() -> None:
    windows = segment_signal(np.arange(10), window_size=4, overlap=2)

    np.testing.assert_array_equal(
        windows,
        np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]]),
    )


def test_segment_signal_can_pad_final_window() -> None:
    windows = segment_signal(
        np.arange(5), window_size=4, overlap=0, drop_incomplete=False
    )

    assert windows.shape == (2, 4)
    np.testing.assert_array_equal(windows[0], [0, 1, 2, 3])
    assert windows[1, 0] == 4
    assert np.all(np.isnan(windows[1, 1:]))

