"""
Tests for Monte Carlo module.
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from src.monte_carlo import generate_monte_carlo_confidence_band


def test_mc_band_shape():
    """Test Monte Carlo band has correct shape."""
    T = 252
    band = generate_monte_carlo_confidence_band(
        T=T,
        sigma_true=0.01,
        halflife=126,
        num_trials=100,  # Small for speed
        seed=42
    )
    
    assert band['P10'].shape == (T,), "P10 should have length T"
    assert band['P50'].shape == (T,), "P50 should have length T"
    assert band['P90'].shape == (T,), "P90 should have length T"


def test_mc_band_percentile_ordering():
    """Test percentiles are correctly ordered."""
    T = 252
    band = generate_monte_carlo_confidence_band(
        T=T,
        sigma_true=0.01,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    # P10 <= P50 <= P90 for all t
    assert np.all(band['P10'] <= band['P50']), "P10 should be <= P50"
    assert np.all(band['P50'] <= band['P90']), "P50 should be <= P90"


def test_mc_band_converges_to_true_vol():
    """Test MC band median converges to true volatility."""
    T = 1000
    sigma_true = 0.01
    
    band = generate_monte_carlo_confidence_band(
        T=T,
        sigma_true=sigma_true,
        halflife=126,
        num_trials=1000,
        seed=42
    )
    
    # Final median should be close to true annualized vol
    true_vol_annual = sigma_true * np.sqrt(252)
    final_median = band['P50'][-1]
    
    # Allow 5% relative error
    assert np.abs(final_median - true_vol_annual) / true_vol_annual < 0.05, \
        "Median should converge to true vol"


def test_mc_band_width_decreases():
    """Test MC band width decreases over time (uncertainty reduces)."""
    T = 500
    band = generate_monte_carlo_confidence_band(
        T=T,
        sigma_true=0.01,
        halflife=126,
        num_trials=500,
        seed=42
    )
    
    # Band width at start vs. end
    width_start = band['P90'][10] - band['P10'][10]  # Day 10
    width_end = band['P90'][-1] - band['P10'][-1]    # Final day
    
    # Width should decrease (less uncertainty with more data)
    assert width_end < width_start, "Band width should decrease over time"


def test_mc_band_reproducibility():
    """Test MC band is reproducible with same seed."""
    T = 252
    sigma_true = 0.01
    
    band1 = generate_monte_carlo_confidence_band(
        T=T, sigma_true=sigma_true, halflife=126, num_trials=100, seed=42
    )
    
    band2 = generate_monte_carlo_confidence_band(
        T=T, sigma_true=sigma_true, halflife=126, num_trials=100, seed=42
    )
    
    # Should be identical
    assert np.allclose(band1['P10'], band2['P10']), "Band should be reproducible"
    assert np.allclose(band1['P50'], band2['P50']), "Band should be reproducible"
    assert np.allclose(band1['P90'], band2['P90']), "Band should be reproducible"


def test_mc_band_different_seeds():
    """Test MC band differs with different seeds."""
    T = 252
    sigma_true = 0.01
    
    band1 = generate_monte_carlo_confidence_band(
        T=T, sigma_true=sigma_true, halflife=126, num_trials=100, seed=42
    )
    
    band2 = generate_monte_carlo_confidence_band(
        T=T, sigma_true=sigma_true, halflife=126, num_trials=100, seed=43
    )
    
    # Should be different
    assert not np.allclose(band1['P10'], band2['P10']), "Different seeds should give different bands"


def test_mc_band_positive_values():
    """Test MC band values are all positive."""
    T = 252
    band = generate_monte_carlo_confidence_band(
        T=T,
        sigma_true=0.01,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    # All percentiles should be positive (volatility is non-negative)
    assert np.all(band['P10'] >= 0), "P10 should be non-negative"
    assert np.all(band['P50'] >= 0), "P50 should be non-negative"
    assert np.all(band['P90'] >= 0), "P90 should be non-negative"


def test_mc_band_true_vol_stored():
    """Test true volatility is stored in result."""
    sigma_true = 0.01
    band = generate_monte_carlo_confidence_band(
        T=100,
        sigma_true=sigma_true,
        halflife=126,
        num_trials=100,
        seed=42
    )
    
    expected = sigma_true * np.sqrt(252)
    assert np.isclose(band['true_vol'], expected), "True vol should be stored"


def test_mc_band_small_T():
    """Test MC band with small T."""
    T = 10
    band = generate_monte_carlo_confidence_band(
        T=T,
        sigma_true=0.01,
        halflife=126,
        num_trials=50,
        seed=42
    )
    
    assert band['P10'].shape == (T,), "Should handle small T"
    assert np.all(band['P10'] <= band['P90']), "Ordering should hold for small T"


def test_mc_band_large_halflife():
    """Test MC band with large halflife relative to T."""
    T = 100
    halflife = 200  # Larger than T
    
    band = generate_monte_carlo_confidence_band(
        T=T,
        sigma_true=0.01,
        halflife=halflife,
        num_trials=50,
        seed=42
    )
    
    # Should still work, but estimates will be slow to converge
    assert band['P10'].shape == (T,), "Should handle large halflife"


def test_mc_band_statistical_coverage():
    """Test that P10-P90 band contains ~80% of estimates."""
    # This is a statistical test - we generate one long series and check coverage
    T = 500
    sigma_true = 0.01
    
    # Generate the band
    band = generate_monte_carlo_confidence_band(
        T=T,
        sigma_true=sigma_true,
        halflife=126,
        num_trials=1000,
        seed=42
    )
    
    # Generate one test series
    np.random.seed(100)
    test_returns = np.random.normal(0, sigma_true, T)
    
    # Compute EWMA vol
    from src.ewma import ewma_volatility_recursive, annualize_volatility
    test_sigma_hat, _ = ewma_volatility_recursive(test_returns, halflife=126)
    test_sigma_ann = annualize_volatility(test_sigma_hat)
    
    # Check coverage in later period (after convergence)
    in_band = (test_sigma_ann[100:] >= band['P10'][100:]) & (test_sigma_ann[100:] <= band['P90'][100:])
    coverage = np.mean(in_band)
    
    # Should be around 80% (10-90 percentiles)
    # Allow wide margin due to single sample
    assert 0.5 < coverage < 0.95, f"Coverage {coverage:.2f} outside expected range"


if __name__ == "__main__":
    print("Running Monte Carlo tests...")
    
    test_mc_band_shape()
    print("✓ test_mc_band_shape")
    
    test_mc_band_percentile_ordering()
    print("✓ test_mc_band_percentile_ordering")
    
    test_mc_band_converges_to_true_vol()
    print("✓ test_mc_band_converges_to_true_vol")
    
    test_mc_band_width_decreases()
    print("✓ test_mc_band_width_decreases")
    
    test_mc_band_reproducibility()
    print("✓ test_mc_band_reproducibility")
    
    test_mc_band_different_seeds()
    print("✓ test_mc_band_different_seeds")
    
    test_mc_band_positive_values()
    print("✓ test_mc_band_positive_values")
    
    test_mc_band_true_vol_stored()
    print("✓ test_mc_band_true_vol_stored")
    
    test_mc_band_small_T()
    print("✓ test_mc_band_small_T")
    
    test_mc_band_large_halflife()
    print("✓ test_mc_band_large_halflife")
    
    test_mc_band_statistical_coverage()
    print("✓ test_mc_band_statistical_coverage")
    
    print("\nAll Monte Carlo tests passed!")
