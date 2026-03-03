"""
Tests for EWMA volatility estimation module.
"""

import numpy as np
import pytest
from src.ewma import (
    compute_ewma_beta,
    compute_ewma_volatility_series,
    compute_ewma_ma_series,
    EWMAVolatilityEstimator,
    EWMAMovingAverage
)


def test_compute_ewma_beta():
    """Test EWMA beta computation."""
    # Halflife of 126 days
    beta = compute_ewma_beta(126.0)
    
    # Beta should be between 0 and 1
    assert 0 < beta < 1
    
    # For halflife h, we should have beta^h ≈ 0.5
    assert np.abs(beta**126 - 0.5) < 0.01
    
    # Test edge cases
    beta_short = compute_ewma_beta(1.0)
    assert beta_short < beta  # Shorter halflife = smaller beta
    
    beta_long = compute_ewma_beta(252.0)
    assert beta_long > beta  # Longer halflife = larger beta


def test_ewma_volatility_constant_returns():
    """Test EWMA volatility with constant returns."""
    # Generate constant returns (zero)
    returns = np.zeros(100)
    
    sigma_hat = compute_ewma_volatility_series(returns, halflife=126.0)
    
    # All estimates should be zero
    assert np.allclose(sigma_hat, 0.0)


def test_ewma_volatility_known_variance():
    """Test EWMA volatility with known variance."""
    # Generate returns with known volatility
    np.random.seed(42)
    true_sigma = 0.01
    returns = np.random.normal(0, true_sigma, 1000)
    
    sigma_hat = compute_ewma_volatility_series(returns, halflife=126.0)
    
    # After many observations, estimate should converge to true value
    final_estimate = sigma_hat[-1]
    assert np.abs(final_estimate - true_sigma) < 0.002  # Within 20%


def test_ewma_volatility_increasing_over_time():
    """Test that EWMA adapts to changing volatility."""
    # Low vol period followed by high vol period
    low_vol_returns = np.random.normal(0, 0.005, 500)
    high_vol_returns = np.random.normal(0, 0.02, 500)
    returns = np.concatenate([low_vol_returns, high_vol_returns])
    
    sigma_hat = compute_ewma_volatility_series(returns, halflife=126.0)
    
    # Volatility estimate should increase in second half
    assert sigma_hat[-1] > sigma_hat[499]


def test_ewma_volatility_positive():
    """Test that EWMA volatility is always non-negative."""
    np.random.seed(123)
    returns = np.random.normal(0, 0.01, 252)
    
    sigma_hat = compute_ewma_volatility_series(returns, halflife=126.0)
    
    # All estimates should be non-negative
    assert np.all(sigma_hat >= 0)


def test_ewma_ma_series():
    """Test EWMA moving average."""
    # Test with constant values
    values = np.ones(100) * 10.0
    ma = compute_ewma_ma_series(values, halflife=20.0)
    
    # MA should converge to constant value
    assert np.abs(ma[-1] - 10.0) < 0.01


def test_ewma_ma_series_trend():
    """Test EWMA MA with trending values."""
    # Linear trend
    values = np.arange(1, 101)
    ma = compute_ewma_ma_series(values, halflife=20.0)
    
    # MA should be increasing
    assert np.all(np.diff(ma) > 0)
    
    # MA should lag behind actual values (smoothing effect)
    assert ma[-1] < values[-1]


def test_ewma_volatility_estimator_online():
    """Test online EWMA volatility estimator."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 100)
    
    # Compute using batch method
    sigma_hat_batch = compute_ewma_volatility_series(returns, halflife=126.0)
    
    # Compute using online estimator
    estimator = EWMAVolatilityEstimator(halflife=126.0)
    sigma_hat_online = []
    for r in returns:
        sigma_hat_online.append(estimator.update(r))
    sigma_hat_online = np.array(sigma_hat_online)
    
    # Results should match
    assert np.allclose(sigma_hat_batch, sigma_hat_online, rtol=1e-10)


def test_ewma_volatility_estimator_reset():
    """Test that reset works correctly."""
    estimator = EWMAVolatilityEstimator(halflife=126.0)
    
    # Update with some data
    for _ in range(10):
        estimator.update(0.01)
    
    assert estimator.k == 10
    assert estimator.S > 0
    
    # Reset
    estimator.reset()
    
    assert estimator.k == 0
    assert estimator.S == 0.0


def test_ewma_ma_online():
    """Test online EWMA moving average."""
    values = np.arange(1.0, 51.0)
    
    # Batch
    ma_batch = compute_ewma_ma_series(values, halflife=20.0)
    
    # Online
    estimator = EWMAMovingAverage(halflife=20.0)
    ma_online = []
    for v in values:
        ma_online.append(estimator.update(v))
    ma_online = np.array(ma_online)
    
    # Should match
    assert np.allclose(ma_batch, ma_online, rtol=1e-10)


def test_ewma_single_observation():
    """Test EWMA with single observation."""
    returns = np.array([0.02])
    
    sigma_hat = compute_ewma_volatility_series(returns, halflife=126.0)
    
    # With one observation, estimate equals absolute value
    assert sigma_hat[0] == 0.02


def test_ewma_normalization_factor():
    """Test that normalization factor converges to 1."""
    # Generate long series
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 5000)
    
    estimator = EWMAVolatilityEstimator(halflife=126.0)
    
    # After many observations, beta^k -> 0, so normalization -> 1
    for r in returns:
        estimator.update(r)
    
    # For k >> halflife, 1-beta^k ≈ 1
    beta_to_k = estimator.beta ** estimator.k
    assert beta_to_k < 1e-10  # Essentially zero


def test_ewma_extreme_values():
    """Test EWMA with extreme return values."""
    # Include some extreme returns
    returns = np.array([0.001, 0.002, 0.1, -0.1, 0.001, 0.002])
    
    sigma_hat = compute_ewma_volatility_series(returns, halflife=3.0)
    
    # All estimates should be positive and finite
    assert np.all(sigma_hat > 0)
    assert np.all(np.isfinite(sigma_hat))
    
    # Volatility should spike after extreme returns
    assert sigma_hat[2] > sigma_hat[1]
    assert sigma_hat[3] > sigma_hat[2]
