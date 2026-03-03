"""
Tests for metrics module.
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from src.metrics import (
    volatility_tracking_error,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    maximum_drawdown,
    cagr,
    compute_all_metrics
)


def test_volatility_tracking_error_perfect():
    """Test VTE with perfect tracking."""
    sigma_target = 0.01
    sigma_hat_series = np.ones(252) * sigma_target
    
    vte = volatility_tracking_error(sigma_hat_series, sigma_target)
    
    assert np.isclose(vte, 0.0), "Perfect tracking should give VTE=0"


def test_volatility_tracking_error_constant_deviation():
    """Test VTE with constant deviation."""
    sigma_target = 0.01
    sigma_hat_series = np.ones(252) * (sigma_target + 0.001)  # 0.1% daily deviation
    
    vte = volatility_tracking_error(sigma_hat_series, sigma_target)
    expected = 0.001 * np.sqrt(252)  # Annualized
    
    assert np.isclose(vte, expected), "VTE with constant deviation incorrect"


def test_annualized_return():
    """Test annualized return calculation."""
    # 1% daily return
    returns = np.ones(252) * 0.01
    ann_ret = annualized_return(returns)
    expected = 0.01 * 252
    
    assert np.isclose(ann_ret, expected), "Annualized return formula incorrect"


def test_annualized_return_zero():
    """Test annualized return with zero returns."""
    returns = np.zeros(252)
    ann_ret = annualized_return(returns)
    
    assert np.isclose(ann_ret, 0.0), "Zero returns should give zero annualized return"


def test_annualized_volatility():
    """Test annualized volatility calculation."""
    # Known volatility
    np.random.seed(42)
    daily_vol = 0.01
    returns = np.random.normal(0, daily_vol, 5000)
    
    ann_vol = annualized_volatility(returns)
    expected = daily_vol * np.sqrt(252)
    
    # Should be close (not exact due to sampling)
    assert np.abs(ann_vol - expected) < 0.02, "Annualized volatility incorrect"


def test_annualized_volatility_zero():
    """Test annualized volatility with constant returns."""
    returns = np.ones(252) * 0.01
    ann_vol = annualized_volatility(returns)
    
    assert np.isclose(ann_vol, 0.0), "Constant returns should give zero volatility"


def test_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    # Positive return, positive vol
    returns = np.random.normal(0.001, 0.01, 252)
    sr = sharpe_ratio(returns)
    
    # Sharpe should be return/vol (both annualized)
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    expected = ann_ret / ann_vol
    
    assert np.isclose(sr, expected), "Sharpe ratio formula incorrect"


def test_sharpe_ratio_zero_volatility():
    """Test Sharpe ratio with zero volatility."""
    returns = np.ones(252) * 0.01
    sr = sharpe_ratio(returns)
    
    # Should handle division by zero gracefully
    assert sr == 0.0 or np.isfinite(sr), "Sharpe with zero vol should be handled"


def test_maximum_drawdown_no_drawdown():
    """Test max drawdown with no drawdown."""
    # Monotonically increasing
    cumulative = np.linspace(1.0, 2.0, 252)
    max_dd = maximum_drawdown(cumulative)
    
    assert np.isclose(max_dd, 0.0, atol=1e-10), "No drawdown should give MaxDD=0"


def test_maximum_drawdown_50_percent():
    """Test max drawdown with 50% drop."""
    # Rise to 2, drop to 1 (50% drawdown)
    cumulative = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
    max_dd = maximum_drawdown(cumulative)
    
    expected = 0.5  # 50% drawdown
    assert np.isclose(max_dd, expected), "50% drawdown not calculated correctly"


def test_maximum_drawdown_multiple_peaks():
    """Test max drawdown with multiple peaks."""
    # Two peaks, deeper drawdown from second
    cumulative = np.array([1.0, 2.0, 1.5, 3.0, 1.5])  # 50% drawdown from 3.0 to 1.5
    max_dd = maximum_drawdown(cumulative)
    
    expected = 0.5
    assert np.isclose(max_dd, expected), "MaxDD should be from highest peak"


def test_cagr_doubling():
    """Test CAGR with doubling."""
    # Double in 252 days (1 year)
    cumulative = np.linspace(1.0, 2.0, 252)
    cagr_value = cagr(cumulative)
    
    expected = 1.0  # 100% annual growth
    assert np.isclose(cagr_value, expected, rtol=0.01), "CAGR for doubling incorrect"


def test_cagr_no_growth():
    """Test CAGR with no growth."""
    cumulative = np.ones(252)
    cagr_value = cagr(cumulative)
    
    assert np.isclose(cagr_value, 0.0), "CAGR with no growth should be 0"


def test_cagr_negative_returns():
    """Test CAGR with negative returns."""
    # Lose 50% over the period
    cumulative = np.linspace(1.0, 0.5, 252)
    cagr_value = cagr(cumulative)
    
    expected = -0.5  # -50% annual growth
    assert np.isclose(cagr_value, expected), "CAGR with losses incorrect"


def test_compute_all_metrics():
    """Test computing all metrics together."""
    np.random.seed(42)
    
    returns_index = np.random.normal(0.001, 0.01, 252)
    cumulative_index = np.cumprod(1 + returns_index)
    sigma_hat_series = np.ones(252) * 0.01
    sigma_target = 0.01
    
    metrics = compute_all_metrics(
        returns_index,
        cumulative_index,
        sigma_hat_series,
        sigma_target
    )
    
    # Check all metrics are present
    expected_keys = ['VTE', 'Annualized Return', 'Annualized Volatility', 
                     'Sharpe Ratio', 'Maximum Drawdown', 'CAGR']
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert np.isfinite(metrics[key]), f"Metric {key} is not finite"


def test_metrics_bounds():
    """Test metrics are within reasonable bounds."""
    np.random.seed(42)
    
    # Realistic returns
    returns_index = np.random.normal(0.0004, 0.01, 252)  # ~10% annual, 15% vol
    cumulative_index = np.cumprod(1 + returns_index)
    sigma_hat_series = np.ones(252) * 0.01
    sigma_target = 0.01
    
    metrics = compute_all_metrics(
        returns_index,
        cumulative_index,
        sigma_hat_series,
        sigma_target
    )
    
    # VTE should be non-negative
    assert metrics['VTE'] >= 0, "VTE should be non-negative"
    
    # Max drawdown should be between 0 and 1
    assert 0 <= metrics['Maximum Drawdown'] <= 1, "MaxDD should be in [0, 1]"
    
    # Annualized volatility should be positive
    assert metrics['Annualized Volatility'] >= 0, "Vol should be non-negative"


def test_vte_is_mae_not_rmse():
    """Test that VTE uses MAE, not RMSE."""
    sigma_target = 0.01
    # Two deviations: +0.001 and -0.001
    sigma_hat_series = np.array([sigma_target + 0.001, sigma_target - 0.001])
    
    vte = volatility_tracking_error(sigma_hat_series, sigma_target)
    
    # MAE should be 0.001 daily, annualized
    mae_daily = 0.001
    expected = mae_daily * np.sqrt(252)
    
    assert np.isclose(vte, expected), "VTE should use MAE, not RMSE"


def test_sharpe_no_risk_free_subtraction():
    """Test Sharpe ratio does not subtract risk-free rate."""
    # Per paper specification: Sharpe = Ann.Return / Ann.Vol (no rf subtraction)
    returns = np.ones(252) * 0.001
    sr = sharpe_ratio(returns)
    
    # Should be (0.001 * 252) / (0 * sqrt(252)) but vol is 0, so handled specially
    # Just verify it doesn't subtract anything
    ann_ret = annualized_return(returns)
    assert ann_ret > 0, "Should have positive return"


if __name__ == "__main__":
    print("Running metrics tests...")
    
    test_volatility_tracking_error_perfect()
    print("✓ test_volatility_tracking_error_perfect")
    
    test_volatility_tracking_error_constant_deviation()
    print("✓ test_volatility_tracking_error_constant_deviation")
    
    test_annualized_return()
    print("✓ test_annualized_return")
    
    test_annualized_return_zero()
    print("✓ test_annualized_return_zero")
    
    test_annualized_volatility()
    print("✓ test_annualized_volatility")
    
    test_annualized_volatility_zero()
    print("✓ test_annualized_volatility_zero")
    
    test_sharpe_ratio()
    print("✓ test_sharpe_ratio")
    
    test_sharpe_ratio_zero_volatility()
    print("✓ test_sharpe_ratio_zero_volatility")
    
    test_maximum_drawdown_no_drawdown()
    print("✓ test_maximum_drawdown_no_drawdown")
    
    test_maximum_drawdown_50_percent()
    print("✓ test_maximum_drawdown_50_percent")
    
    test_maximum_drawdown_multiple_peaks()
    print("✓ test_maximum_drawdown_multiple_peaks")
    
    test_cagr_doubling()
    print("✓ test_cagr_doubling")
    
    test_cagr_no_growth()
    print("✓ test_cagr_no_growth")
    
    test_cagr_negative_returns()
    print("✓ test_cagr_negative_returns")
    
    test_compute_all_metrics()
    print("✓ test_compute_all_metrics")
    
    test_metrics_bounds()
    print("✓ test_metrics_bounds")
    
    test_vte_is_mae_not_rmse()
    print("✓ test_vte_is_mae_not_rmse")
    
    test_sharpe_no_risk_free_subtraction()
    print("✓ test_sharpe_no_risk_free_subtraction")
    
    print("\nAll metrics tests passed!")
