"""
Tests for EWMA module.
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from src.ewma import (
    compute_ewma_beta,
    ewma_volatility_recursive,
    ewma_moving_average_recursive,
    annualize_volatility,
    compute_target_daily_volatility,
    EWMAVolatilityEstimator,
    EWMAMovingAverage
)


def test_compute_ewma_beta():
    """Test EWMA beta computation."""
    # Test with halflife = 126
    beta = compute_ewma_beta(126)
    assert 0 < beta < 1, "Beta should be between 0 and 1"
    assert np.isclose(beta, np.exp(-np.log(2.0) / 126)), "Beta formula incorrect"
    
    # Test with halflife = 1 (should give beta close to 0.5)
    beta_1 = compute_ewma_beta(1)
    assert np.isclose(beta_1, 0.5), "Beta with halflife=1 should be ~0.5"


def test_ewma_volatility_constant_returns():
    """Test EWMA volatility with constant returns."""
    # Constant returns should give constant volatility estimate
    returns = np.ones(100) * 0.01
    sigma_hat, S = ewma_volatility_recursive(returns, halflife=126)
    
    # All estimates should be very close to 0.01
    assert np.allclose(sigma_hat, 0.01, rtol=0.05), "Constant returns should give constant vol"


def test_ewma_volatility_convergence():
    """Test EWMA volatility converges to true volatility."""
    np.random.seed(42)
    true_vol = 0.02
    returns = np.random.normal(0, true_vol, 5000)
    
    sigma_hat, S = ewma_volatility_recursive(returns, halflife=126)
    
    # Final estimate should be close to true vol
    assert np.abs(sigma_hat[-1] - true_vol) < 0.005, "EWMA should converge to true vol"


def test_ewma_volatility_online_matches_batch():
    """Test online estimator matches batch computation."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 500)
    
    # Batch computation
    sigma_hat_batch, _ = ewma_volatility_recursive(returns, halflife=126)
    
    # Online computation
    estimator = EWMAVolatilityEstimator(halflife=126)
    sigma_hat_online = np.array([estimator.update(r) for r in returns])
    
    assert np.allclose(sigma_hat_batch, sigma_hat_online), "Online should match batch"


def test_ewma_moving_average():
    """Test EWMA moving average."""
    # Constant level should give constant MA
    levels = np.ones(100) * 10.0
    ma = ewma_moving_average_recursive(levels, halflife=50)
    
    # All MA values should converge to 10
    assert np.allclose(ma[-50:], 10.0, rtol=0.01), "MA of constant should be constant"


def test_annualize_volatility():
    """Test volatility annualization."""
    daily_vol = 0.01
    annual_vol = annualize_volatility(np.array([daily_vol]))
    expected = daily_vol * np.sqrt(252)
    
    assert np.isclose(annual_vol[0], expected), "Annualization formula incorrect"


def test_compute_target_daily_volatility():
    """Test target daily volatility computation."""
    annual_target = 0.15
    daily_target = compute_target_daily_volatility(annual_target)
    expected = 0.15 / np.sqrt(252)
    
    assert np.isclose(daily_target, expected), "Daily target formula incorrect"


def test_ewma_volatility_zero_returns():
    """Test EWMA with zero returns edge case."""
    returns = np.zeros(100)
    sigma_hat, S = ewma_volatility_recursive(returns, halflife=126)
    
    # All volatility estimates should be zero
    assert np.allclose(sigma_hat, 0), "Zero returns should give zero vol"


def test_ewma_volatility_single_observation():
    """Test EWMA with single observation."""
    returns = np.array([0.02])
    sigma_hat, S = ewma_volatility_recursive(returns, halflife=126)
    
    # Single observation should give vol equal to abs(return)
    assert len(sigma_hat) == 1, "Should have one estimate"
    assert np.isclose(sigma_hat[0], 0.02), "Single observation vol should be |return|"


def test_ewma_ma_online_matches_batch():
    """Test online MA matches batch computation."""
    np.random.seed(42)
    levels = np.cumsum(np.random.normal(0, 1, 500))
    
    # Batch
    ma_batch = ewma_moving_average_recursive(levels, halflife=50)
    
    # Online
    estimator = EWMAMovingAverage(halflife=50)
    ma_online = np.array([estimator.update(l) for l in levels])
    
    assert np.allclose(ma_batch, ma_online), "Online MA should match batch"


def test_ewma_volatility_positive():
    """Test EWMA volatility is always positive."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 1000)
    returns[500] = -0.05  # Large negative return
    returns[600] = 0.05   # Large positive return
    
    sigma_hat, S = ewma_volatility_recursive(returns, halflife=126)
    
    assert np.all(sigma_hat >= 0), "Volatility should always be non-negative"


def test_ewma_normalization_factor():
    """Test EWMA normalization factor is computed correctly."""
    # The normalization factor 1 - beta^k should approach 1 as k increases
    halflife = 126
    beta = compute_ewma_beta(halflife)
    
    # For large k, beta^k should be very small
    k_large = 1000
    beta_power = beta ** k_large
    assert beta_power < 0.01, "Beta^k should be small for large k"
    
    # Therefore, 1 - beta^k should be close to 1
    norm_factor = 1 - beta_power
    assert np.isclose(norm_factor, 1.0, rtol=0.01), "Norm factor should approach 1"


if __name__ == "__main__":
    print("Running EWMA tests...")
    
    test_compute_ewma_beta()
    print("✓ test_compute_ewma_beta")
    
    test_ewma_volatility_constant_returns()
    print("✓ test_ewma_volatility_constant_returns")
    
    test_ewma_volatility_convergence()
    print("✓ test_ewma_volatility_convergence")
    
    test_ewma_volatility_online_matches_batch()
    print("✓ test_ewma_volatility_online_matches_batch")
    
    test_ewma_moving_average()
    print("✓ test_ewma_moving_average")
    
    test_annualize_volatility()
    print("✓ test_annualize_volatility")
    
    test_compute_target_daily_volatility()
    print("✓ test_compute_target_daily_volatility")
    
    test_ewma_volatility_zero_returns()
    print("✓ test_ewma_volatility_zero_returns")
    
    test_ewma_volatility_single_observation()
    print("✓ test_ewma_volatility_single_observation")
    
    test_ewma_ma_online_matches_batch()
    print("✓ test_ewma_ma_online_matches_batch")
    
    test_ewma_volatility_positive()
    print("✓ test_ewma_volatility_positive")
    
    test_ewma_normalization_factor()
    print("✓ test_ewma_normalization_factor")
    
    print("\nAll EWMA tests passed!")
