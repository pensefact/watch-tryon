import math
import numpy as np
import pytest

from src.video.tracker import OneEuroFilter


def test_filter_returns_first_value():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    result = f(0.0, 100.0)
    assert abs(result - 100.0) < 0.01


def test_filter_smooths_noise():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    values = [100.0, 102.0, 98.0, 101.0, 99.0, 100.5, 100.0]
    filtered = []
    for i, v in enumerate(values):
        filtered.append(f(i * 0.033, v))  # 30fps

    # Filtered values should have less variance than input
    input_var = np.var(values)
    filtered_var = np.var(filtered)
    assert filtered_var < input_var


def test_filter_tracks_large_changes():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    # Start at 100, jump to 200
    f(0.0, 100.0)
    result = f(0.033, 200.0)
    # With beta > 0, the filter should follow large changes more aggressively
    assert result > 150.0, f"Filter should track large jump, got {result}"


def test_filter_independent_per_instance():
    f1 = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f2 = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f1(0.0, 100.0)
    f2(0.0, 200.0)
    assert abs(f1(0.033, 105.0) - f2(0.033, 205.0)) < 1.0
