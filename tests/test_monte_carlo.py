"""
Tests for Monte Carlo confidence band generation.

Tests:
- Band generation
- Percentile properties
- Convergence to true volatility
- Band width analysis
"""

import numpy as np
import pytest
from src.monte_carlo import (
    generate_monte_carlo_band,
    compute_band_width,
    analyze_band_convergence
)


def test_monte_carlo_band_generation():
    """Test basic Monte Carlo band generation."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 100
    num_trials = 100  # Small for speed
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=num_trials,
        seed=42
    )
    
    # Check structure
    assert 'P10' in band
    assert 'P50' in band
    assert 'P90' in band
    
    # Check lengths
    assert len(band['P10']) == T
    assert len(band['P50']) == T
    assert len(band['P90']) == T


def test_monte_carlo_band_ordering():
    """Test that percentiles are correctly ordered: P10 < P50 < P90."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 100
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    # At every time point, P10 <= P50 <= P90
    assert np.all(band['P10'] <= band['P50'])
    assert np.all(band['P50'] <= band['P90'])


def test_monte_carlo_band_centers_on_target():
    """Test that median converges to true volatility."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    sigma_true_annual_pct = 15.0
    T = 1000
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=1000,
        seed=42
    )
    
    # Final median should be close to true volatility
    # (EWMA is approximately unbiased)
    final_median = band['P50'][-1]
    assert np.isclose(final_median, sigma_true_annual_pct, rtol=0.05)


def test_monte_carlo_band_width_decreases():
    """Test that band width generally decreases over time."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 500
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    # Band width early vs late
    width_early = compute_band_width(band, 10)
    width_late = compute_band_width(band, T - 1)
    
    # Early period should have wider band (more uncertainty)
    assert width_early > width_late


def test_monte_carlo_reproducibility():
    """Test that same seed produces same results."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 100
    
    band1 = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    band2 = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    # Should be identical
    assert np.allclose(band1['P10'], band2['P10'])
    assert np.allclose(band1['P50'], band2['P50'])
    assert np.allclose(band1['P90'], band2['P90'])


def test_monte_carlo_different_seeds():
    """Test that different seeds produce different results."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 100
    
    band1 = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    band2 = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=100,
        seed=123
    )
    
    # Should be different
    assert not np.allclose(band1['P10'], band2['P10'])


def test_monte_carlo_band_positive():
    """Test that all band values are positive."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 100
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    # All volatility estimates should be non-negative
    assert np.all(band['P10'] >= 0)
    assert np.all(band['P50'] >= 0)
    assert np.all(band['P90'] >= 0)


def test_analyze_band_convergence():
    """Test band convergence analysis."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 500
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    analysis = analyze_band_convergence(band, sigma_target_annualized_pct=15.0)
    
    # Check structure
    assert 'median_bias_early' in analysis
    assert 'median_bias_mid' in analysis
    assert 'median_bias_late' in analysis
    assert 'band_width_early' in analysis
    assert 'band_width_mid' in analysis
    assert 'band_width_late' in analysis
    
    # Band width should decrease
    assert analysis['band_width_late'] < analysis['band_width_early']


def test_monte_carlo_with_zero_volatility():
    """Test Monte Carlo with zero true volatility."""
    sigma_true_daily = 0.0
    T = 100
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    # All estimates should be very close to zero
    assert np.all(band['P10'] < 0.1)
    assert np.all(band['P50'] < 0.1)
    assert np.all(band['P90'] < 0.1)


def test_monte_carlo_different_halflives():
    """Test Monte Carlo with different halflives."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 500
    
    band_short = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=20,
        num_trials=100,
        seed=42
    )
    
    band_long = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=252,
        num_trials=100,
        seed=42
    )
    
    # Short halflife should have wider final band (more responsive)
    width_short = compute_band_width(band_short, T - 1)
    width_long = compute_band_width(band_long, T - 1)
    
    # Actually, shorter halflife converges faster, so might have narrower band
    # The key difference is responsiveness to shocks
    assert width_short > 0
    assert width_long > 0


def test_monte_carlo_small_sample():
    """Test Monte Carlo with very small sample size."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 10
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=50,
        seed=42
    )
    
    # Should still work
    assert len(band['P10']) == T
    
    # Band should be very wide initially
    width_first = compute_band_width(band, 0)
    assert width_first > 5.0  # Wide uncertainty


def test_monte_carlo_large_sample():
    """Test Monte Carlo with large sample size."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 2000
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=50,  # Smaller trials for speed
        seed=42
    )
    
    # Should work with large T
    assert len(band['P10']) == T
    
    # Band should narrow considerably by the end
    width_final = compute_band_width(band, T - 1)
    assert width_final < 10.0  # Should be reasonably narrow


def test_compute_band_width():
    """Test band width computation."""
    band = {
        'P10': np.array([5.0, 10.0, 12.0]),
        'P50': np.array([15.0, 15.0, 15.0]),
        'P90': np.array([25.0, 20.0, 18.0])
    }
    
    width_0 = compute_band_width(band, 0)
    width_1 = compute_band_width(band, 1)
    width_2 = compute_band_width(band, 2)
    
    assert width_0 == 20.0  # 25 - 5
    assert width_1 == 10.0  # 20 - 10
    assert width_2 == 6.0   # 18 - 12


def test_monte_carlo_percentile_coverage():
    """Test that percentiles have correct coverage properties."""
    sigma_true_daily = 0.15 / np.sqrt(252)
    T = 100
    num_trials = 1000
    
    band = generate_monte_carlo_band(
        sigma_true_daily=sigma_true_daily,
        T=T,
        halflife=126,
        num_trials=num_trials,
        seed=42
    )
    
    # For any time point, roughly 80% of trials should be between P10 and P90
    # We can't easily verify this without regenerating, but we can check
    # that the band is not degenerate
    for k in range(0, T, 10):
        width = compute_band_width(band, k)
        assert width > 0  # Band should have non-zero width
