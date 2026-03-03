"""
Tests for EWMA volatility estimation module.

Tests:
- Recursive computation correctness
- Normalization factor
- Edge cases (single observation, zeros)
- Mathematical properties
"""

import numpy as np
import pytest
from src.ewma import (
    compute_ewma_volatility,
    compute_ewma_volatility_annualized,
    compute_ewma_ma,
    EWMAVolatilityEstimator
)


def test_ewma_single_observation():
    """Test EWMA with single return."""
    returns = np.array([0.01])
    halflife = 126
    
    sigma_hat = compute_ewma_volatility(returns, halflife)
    
    # For single observation: sigma_hat_1 = |r_1|
    assert len(sigma_hat) == 1
    assert np.isclose(sigma_hat[0], abs(returns[0]))


def test_ewma_constant_returns():
    """Test EWMA with constant returns."""
    returns = np.ones(100) * 0.01
    halflife = 126
    
    sigma_hat = compute_ewma_volatility(returns, halflife)
    
    # Should converge to the constant value
    assert np.all(sigma_hat > 0)
    assert np.isclose(sigma_hat[-1], 0.01, atol=1e-6)


def test_ewma_zero_returns():
    """Test EWMA with all zero returns."""
    returns = np.zeros(100)
    halflife = 126
    
    sigma_hat = compute_ewma_volatility(returns, halflife)
    
    # All volatility estimates should be zero
    assert np.all(sigma_hat == 0)


def test_ewma_normalization():
    """Test that EWMA is properly normalized."""
    np.random.seed(42)
    sigma_true = 0.01
    returns = np.random.normal(0, sigma_true, 1000)
    halflife = 126
    
    sigma_hat = compute_ewma_volatility(returns, halflife)
    
    # Final estimate should be close to true volatility
    assert np.isclose(sigma_hat[-1], sigma_true, rtol=0.2)


def test_ewma_annualization():
    """Test annualization factor is applied correctly."""
    returns = np.random.normal(0, 0.01, 100)
    halflife = 126
    
    sigma_daily = compute_ewma_volatility(returns, halflife)
    sigma_annual = compute_ewma_volatility_annualized(returns, halflife)
    
    # Annual should be daily * sqrt(252)
    assert np.allclose(sigma_annual, sigma_daily * np.sqrt(252))


def test_ewma_ma_constant_series():
    """Test EWMA moving average with constant series."""
    series = np.ones(100) * 5.0
    halflife = 126
    
    ma = compute_ewma_ma(series, halflife)
    
    # Should converge to the constant value
    assert np.all(ma > 0)
    assert np.isclose(ma[-1], 5.0, atol=1e-6)


def test_ewma_ma_trending_series():
    """Test EWMA moving average with trending series."""
    series = np.arange(1, 101, dtype=float)
    halflife = 20
    
    ma = compute_ewma_ma(series, halflife)
    
    # MA should be increasing and lag behind the series
    assert np.all(np.diff(ma) > 0)  # Monotonically increasing
    assert ma[-1] < series[-1]  # Lags behind


def test_ewma_estimator_class():
    """Test EWMAVolatilityEstimator class interface."""
    estimator = EWMAVolatilityEstimator(halflife=126)
    
    returns = np.random.normal(0, 0.01, 100)
    sigma_estimates = []
    
    for r in returns:
        sigma = estimator.update(r)
        sigma_estimates.append(sigma)
    
    # Compare with batch computation
    sigma_batch = compute_ewma_volatility(returns, 126)
    
    assert np.allclose(sigma_estimates, sigma_batch)


def test_ewma_estimator_reset():
    """Test reset functionality of EWMAVolatilityEstimator."""
    estimator = EWMAVolatilityEstimator(halflife=126)
    
    # Update with some data
    for _ in range(10):
        estimator.update(0.01)
    
    # Reset
    estimator.reset()
    
    # State should be back to initial
    assert estimator.k == 0
    assert estimator.S == 0.0


def test_ewma_different_halflives():
    """Test that different halflives produce different estimates."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 500)
    
    sigma_short = compute_ewma_volatility(returns, halflife=20)
    sigma_long = compute_ewma_volatility(returns, halflife=252)
    
    # Short halflife should be more responsive (higher variance of estimates)
    variance_short = np.var(sigma_short[100:])
    variance_long = np.var(sigma_long[100:])
    
    assert variance_short > variance_long


def test_ewma_positive_returns_only():
    """Test EWMA with all positive returns."""
    returns = np.abs(np.random.normal(0, 0.01, 100))
    halflife = 126
    
    sigma_hat = compute_ewma_volatility(returns, halflife)
    
    # All estimates should be positive
    assert np.all(sigma_hat >= 0)


def test_ewma_negative_returns_only():
    """Test EWMA with all negative returns."""
    returns = -np.abs(np.random.normal(0, 0.01, 100))
    halflife = 126
    
    sigma_hat = compute_ewma_volatility(returns, halflife)
    
    # Volatility should be same as for positive returns (uses r^2)
    returns_pos = -returns
    sigma_hat_pos = compute_ewma_volatility(returns_pos, halflife)
    
    assert np.allclose(sigma_hat, sigma_hat_pos)


def test_ewma_extreme_values():
    """Test EWMA with extreme return values."""
    returns = np.array([0.0] * 95 + [0.5] + [0.0] * 4)  # Large shock
    halflife = 20
    
    sigma_hat = compute_ewma_volatility(returns, halflife)
    
    # Volatility should spike at shock and decay after
    shock_index = 95
    assert sigma_hat[shock_index] > sigma_hat[shock_index - 1]
    assert sigma_hat[-1] < sigma_hat[shock_index + 1]


def test_ewma_monotonic_decay_after_shock():
    """Test that EWMA decays monotonically after a shock with no new shocks."""
    returns = np.zeros(200)
    returns[100] = 0.1  # Single shock
    halflife = 126
    
    sigma_hat = compute_ewma_volatility(returns, halflife)
    
    # After shock, should decay monotonically
    post_shock = sigma_hat[101:]
    assert np.all(np.diff(post_shock) < 0)  # Strictly decreasing


def test_ewma_beta_calculation():
    """Test that beta is calculated correctly from halflife."""
    halflife = 126
    estimator = EWMAVolatilityEstimator(halflife=halflife)
    
    expected_beta = np.exp(-np.log(2) / halflife)
    assert np.isclose(estimator.beta, expected_beta)


def test_ewma_convergence_to_true_vol():
    """Test that EWMA converges to true volatility in large samples."""
    np.random.seed(123)
    sigma_true = 0.015
    returns = np.random.normal(0, sigma_true, 10000)
    halflife = 126
    
    sigma_hat = compute_ewma_volatility(returns, halflife)
    
    # Average of last 1000 estimates should be close to true vol
    avg_final = np.mean(sigma_hat[-1000:])
    assert np.isclose(avg_final, sigma_true, rtol=0.1)
