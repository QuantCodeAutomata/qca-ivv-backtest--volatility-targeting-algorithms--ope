"""
Tests for Monte Carlo simulation module.
"""

import numpy as np
import pytest
from src.monte_carlo import (
    generate_mc_confidence_band,
    generate_mc_confidence_band_batch
)


def test_mc_band_basic():
    """Test basic Monte Carlo band generation."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 126.0
    n_days = 100
    n_trials = 100  # Small for fast test
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # Check shapes
    assert len(P10) == n_days
    assert len(P50) == n_days
    assert len(P90) == n_days
    
    # Check ordering: P10 <= P50 <= P90
    assert np.all(P10 <= P50)
    assert np.all(P50 <= P90)


def test_mc_band_convergence_to_true_vol():
    """Test that median converges to true volatility."""
    true_sigma = 0.15 / np.sqrt(252)
    true_annual = 15.0  # 15% annualized
    halflife = 126.0
    n_days = 1000
    n_trials = 500
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # Median should be close to true volatility (especially later in series)
    # Allow some tolerance due to finite samples
    assert np.abs(P50[-1] - true_annual) < 2.0


def test_mc_band_width_narrows():
    """Test that confidence band narrows over time."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 126.0
    n_days = 500
    n_trials = 200
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # Band width at start
    early_width = P90[9] - P10[9]  # Day 10
    
    # Band width at end
    late_width = P90[-1] - P10[-1]
    
    # Band should narrow (uncertainty decreases)
    assert late_width < early_width


def test_mc_band_positive_values():
    """Test that all band values are positive."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 126.0
    n_days = 100
    n_trials = 100
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # All volatility estimates should be non-negative
    assert np.all(P10 >= 0)
    assert np.all(P50 >= 0)
    assert np.all(P90 >= 0)


def test_mc_band_reproducibility():
    """Test that results are reproducible with same seed."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 126.0
    n_days = 100
    n_trials = 100
    
    P10_1, P50_1, P90_1 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    P10_2, P50_2, P90_2 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # Results should be identical
    assert np.allclose(P10_1, P10_2)
    assert np.allclose(P50_1, P50_2)
    assert np.allclose(P90_1, P90_2)


def test_mc_band_different_seeds():
    """Test that different seeds produce different results."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 126.0
    n_days = 100
    n_trials = 100
    
    P10_1, P50_1, P90_1 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    P10_2, P50_2, P90_2 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=123
    )
    
    # Results should be different (with high probability)
    assert not np.allclose(P10_1, P10_2)


def test_mc_band_batch_matches_full():
    """Test that batched version matches full version."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 126.0
    n_days = 100
    n_trials = 500
    batch_size = 100
    
    # Full version
    P10_full, P50_full, P90_full = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # Batched version
    P10_batch, P50_batch, P90_batch = generate_mc_confidence_band_batch(
        true_sigma, halflife, n_days, n_trials, batch_size, seed=42
    )
    
    # Results should be very close (identical random state)
    assert np.allclose(P10_full, P10_batch, rtol=1e-10)
    assert np.allclose(P50_full, P50_batch, rtol=1e-10)
    assert np.allclose(P90_full, P90_batch, rtol=1e-10)


def test_mc_band_percentile_coverage():
    """Test that percentiles have correct coverage."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 126.0
    n_days = 50
    n_trials = 1000
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # For any given day, approximately 80% of samples should be in [P10, P90]
    # We can't test this directly without regenerating samples,
    # but we can check that the band is reasonable
    
    # Band should not be zero-width
    assert np.all(P90 > P10)


def test_mc_band_short_halflife():
    """Test MC band with short halflife."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 20.0  # Short halflife
    n_days = 100
    n_trials = 100
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # All percentiles should be valid
    assert np.all(np.isfinite(P10))
    assert np.all(np.isfinite(P50))
    assert np.all(np.isfinite(P90))
    
    # Ordering preserved
    assert np.all(P10 <= P50)
    assert np.all(P50 <= P90)


def test_mc_band_long_halflife():
    """Test MC band with long halflife."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 252.0  # Long halflife
    n_days = 100
    n_trials = 100
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # All percentiles should be valid
    assert np.all(np.isfinite(P10))
    assert np.all(np.isfinite(P50))
    assert np.all(np.isfinite(P90))


def test_mc_band_high_volatility():
    """Test MC band with high true volatility."""
    true_sigma = 0.30 / np.sqrt(252)  # 30% annualized
    halflife = 126.0
    n_days = 100
    n_trials = 100
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # Median should be near 30%
    assert 25.0 < P50[-1] < 35.0


def test_mc_band_low_volatility():
    """Test MC band with low true volatility."""
    true_sigma = 0.05 / np.sqrt(252)  # 5% annualized
    halflife = 126.0
    n_days = 100
    n_trials = 100
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # Median should be near 5%
    assert 3.0 < P50[-1] < 7.0


def test_mc_band_early_uncertainty():
    """Test that early periods have high uncertainty."""
    true_sigma = 0.15 / np.sqrt(252)
    halflife = 126.0
    n_days = 200
    n_trials = 500
    
    P10, P50, P90 = generate_mc_confidence_band(
        true_sigma, halflife, n_days, n_trials, seed=42
    )
    
    # Early band width should be larger
    early_width = P90[4] - P10[4]  # Day 5
    mid_width = P90[99] - P10[99]  # Day 100
    
    assert early_width > mid_width
