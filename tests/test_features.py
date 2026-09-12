"""Tests for deterministic feature definitions on synthetic windows."""

import numpy as np
import pytest

from src.features.emg_features import (
    extract_feature_vector,
    mean_absolute_value,
    mean_frequency,
    median_frequency,
    root_mean_square,
    signal_variance,
    slope_sign_changes,
    waveform_length,
    zero_crossings,
)


def test_time_domain_features_have_expected_values() -> None:
    window = np.array([-1.0, 1.0, -1.0, 1.0])

    assert mean_absolute_value(window) == pytest.approx(1.0)
    assert root_mean_square(window) == pytest.approx(1.0)
    assert waveform_length(window) == pytest.approx(6.0)
    assert signal_variance(window) == pytest.approx(1.0)
    assert zero_crossings(window) == 3
    assert slope_sign_changes(window) == 2


def test_threshold_suppresses_small_zero_crossings() -> None:
    window = np.array([-0.01, 0.01, -1.0, 1.0])

    assert zero_crossings(window, threshold=0.1) == 2


def test_spectral_features_locate_single_tone() -> None:
    sampling_frequency_hz = 1_000.0
    time = np.arange(0, 2, 1 / sampling_frequency_hz)
    window = np.sin(2 * np.pi * 100 * time)

    assert mean_frequency(window, sampling_frequency_hz) == pytest.approx(100.0, abs=0.5)
    assert median_frequency(window, sampling_frequency_hz) == pytest.approx(100.0, abs=0.5)


def test_frequency_features_require_sampling_frequency() -> None:
    with pytest.raises(ValueError, match="sampling_frequency_hz"):
        extract_feature_vector([1.0, -1.0, 1.0], selected_features=["mean_frequency"])


def test_constant_signal_has_undefined_spectral_features() -> None:
    with pytest.raises(ValueError, match="spectral power"):
        mean_frequency(np.ones(100), 1_000.0)


def test_extract_selected_features_preserves_requested_order() -> None:
    features = extract_feature_vector(
        [-1.0, 0.0, 1.0], selected_features=["rms", "mav"]
    )

    assert list(features) == ["rms", "mav"]


def test_multichannel_window_requires_explicit_channel_selection() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        mean_absolute_value(np.ones((10, 2)))

