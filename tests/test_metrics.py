"""
Tests for performance metrics computation.

Tests:
- VTE calculation
- Annualized return/volatility
- Sharpe ratio
- Maximum drawdown
- Edge cases
"""

import numpy as np
import pytest
from src.metrics import (
    compute_volatility_tracking_error,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_maximum_drawdown,
    compute_cagr,
    compute_all_metrics,
    validate_metrics
)


def test_vte_zero_tracking_error():
    """Test VTE when volatility matches target exactly."""
    sigma_target_daily = 0.01
    sigma_hat = np.ones(100) * sigma_target_daily
    
    vte = compute_volatility_tracking_error(sigma_hat, sigma_target_daily)
    
    # Should be zero (in percent)
    assert np.isclose(vte, 0.0)


def test_vte_constant_tracking_error():
    """Test VTE with constant deviation from target."""
    sigma_target_daily = 0.01
    sigma_hat = np.ones(100) * (sigma_target_daily + 0.001)  # +0.001 deviation
    
    vte = compute_volatility_tracking_error(sigma_hat, sigma_target_daily)
    
    # VTE = 0.001 * sqrt(252) * 100
    expected = 0.001 * np.sqrt(252) * 100
    assert np.isclose(vte, expected, rtol=1e-5)


def test_annualized_return_positive():
    """Test annualized return with positive daily returns."""
    returns = np.ones(252) * 0.001  # 0.1% daily
    
    ann_return = compute_annualized_return(returns)
    
    # Expected: 0.001 * 252 * 100 = 25.2%
    expected = 0.001 * 252 * 100
    assert np.isclose(ann_return, expected)


def test_annualized_return_negative():
    """Test annualized return with negative daily returns."""
    returns = np.ones(252) * -0.001  # -0.1% daily
    
    ann_return = compute_annualized_return(returns)
    
    # Expected: -0.001 * 252 * 100 = -25.2%
    expected = -0.001 * 252 * 100
    assert np.isclose(ann_return, expected)


def test_annualized_return_zero():
    """Test annualized return with zero returns."""
    returns = np.zeros(252)
    
    ann_return = compute_annualized_return(returns)
    
    assert np.isclose(ann_return, 0.0)


def test_annualized_volatility_constant():
    """Test annualized volatility with known volatility."""
    np.random.seed(42)
    sigma_daily = 0.01
    returns = np.random.normal(0, sigma_daily, 10000)
    
    ann_vol = compute_annualized_volatility(returns)
    
    # Expected: sigma_daily * sqrt(252) * 100
    expected = sigma_daily * np.sqrt(252) * 100
    assert np.isclose(ann_vol, expected, rtol=0.05)


def test_annualized_volatility_zero():
    """Test annualized volatility with constant returns."""
    returns = np.ones(252) * 0.01
    
    ann_vol = compute_annualized_volatility(returns)
    
    # Volatility of constant returns is zero
    assert np.isclose(ann_vol, 0.0)


def test_sharpe_ratio_calculation():
    """Test Sharpe ratio calculation."""
    ann_return = 10.0  # 10% annual return
    ann_vol = 20.0     # 20% annual volatility
    
    sharpe = compute_sharpe_ratio(ann_return, ann_vol)
    
    # Expected: 10 / 20 = 0.5
    assert np.isclose(sharpe, 0.5)


def test_sharpe_ratio_zero_volatility():
    """Test Sharpe ratio with zero volatility."""
    ann_return = 10.0
    ann_vol = 0.0
    
    sharpe = compute_sharpe_ratio(ann_return, ann_vol)
    
    # Should return 0 (not divide by zero)
    assert sharpe == 0.0


def test_sharpe_ratio_negative_return():
    """Test Sharpe ratio with negative return."""
    ann_return = -5.0
    ann_vol = 15.0
    
    sharpe = compute_sharpe_ratio(ann_return, ann_vol)
    
    # Expected: -5 / 15 = -0.333...
    assert np.isclose(sharpe, -5.0 / 15.0)


def test_maximum_drawdown_monotonic_growth():
    """Test max drawdown with monotonically growing returns."""
    cumulative_returns = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    
    max_dd = compute_maximum_drawdown(cumulative_returns)
    
    # No drawdown
    assert np.isclose(max_dd, 0.0)


def test_maximum_drawdown_simple_case():
    """Test max drawdown with simple drawdown scenario."""
    # Grow to 2.0, then fall to 1.5 (25% drawdown)
    cumulative_returns = np.array([1.0, 1.5, 2.0, 1.8, 1.5])
    
    max_dd = compute_maximum_drawdown(cumulative_returns)
    
    # Max DD = 1 - 1.5/2.0 = 0.25 = 25%
    expected = (1 - 1.5 / 2.0) * 100
    assert np.isclose(max_dd, expected)


def test_maximum_drawdown_multiple_drawdowns():
    """Test max drawdown with multiple drawdown periods."""
    # First DD: 30%, Second DD: 40%
    cumulative_returns = np.array([1.0, 1.5, 1.05, 1.2, 2.0, 1.2, 1.5])
    
    max_dd = compute_maximum_drawdown(cumulative_returns)
    
    # Max DD = 1 - 1.2/2.0 = 0.4 = 40%
    expected = (1 - 1.2 / 2.0) * 100
    assert np.isclose(max_dd, expected)


def test_maximum_drawdown_full_loss():
    """Test max drawdown with complete loss."""
    cumulative_returns = np.array([1.0, 0.5, 0.0])
    
    max_dd = compute_maximum_drawdown(cumulative_returns)
    
    # 100% drawdown
    assert np.isclose(max_dd, 100.0)


def test_cagr_positive_growth():
    """Test CAGR with positive growth."""
    # Double in 252 days (1 year)
    cumulative_returns = np.linspace(1.0, 2.0, 252)
    
    cagr = compute_cagr(cumulative_returns)
    
    # CAGR should be 100%
    assert np.isclose(cagr, 100.0, rtol=0.01)


def test_cagr_negative_growth():
    """Test CAGR with negative growth."""
    # Halve in 252 days
    cumulative_returns = np.linspace(1.0, 0.5, 252)
    
    cagr = compute_cagr(cumulative_returns)
    
    # CAGR should be -50%
    assert np.isclose(cagr, -50.0, rtol=0.01)


def test_cagr_no_growth():
    """Test CAGR with no growth."""
    cumulative_returns = np.ones(252)
    
    cagr = compute_cagr(cumulative_returns)
    
    # CAGR should be 0%
    assert np.isclose(cagr, 0.0)


def test_compute_all_metrics():
    """Test compute_all_metrics function."""
    # Create mock results
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 252)
    cumulative_returns = np.cumprod(1 + returns)
    sigma_hat = np.ones(252) * 0.01
    
    results = {
        'index_returns': returns,
        'cumulative_returns': cumulative_returns,
        'sigma_hat': sigma_hat
    }
    
    sigma_target_daily = 0.01
    
    metrics = compute_all_metrics(results, sigma_target_daily)
    
    # Check all metrics are present
    assert 'VTE' in metrics
    assert 'Annualized_Return' in metrics
    assert 'Annualized_Volatility' in metrics
    assert 'Sharpe_Ratio' in metrics
    assert 'Maximum_Drawdown' in metrics
    assert 'CAGR' in metrics
    
    # Check all metrics are finite
    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite"


def test_validate_metrics_valid():
    """Test validate_metrics with valid metrics."""
    metrics = {
        'VTE': 2.5,
        'Annualized_Return': 10.0,
        'Annualized_Volatility': 15.0,
        'Sharpe_Ratio': 0.67,
        'Maximum_Drawdown': 25.0,
        'CAGR': 9.5
    }
    
    # Should not raise
    validate_metrics(metrics)


def test_validate_metrics_negative_vte():
    """Test validate_metrics with negative VTE."""
    metrics = {
        'VTE': -1.0,
        'Annualized_Return': 10.0,
        'Annualized_Volatility': 15.0,
        'Sharpe_Ratio': 0.67,
        'Maximum_Drawdown': 25.0,
        'CAGR': 9.5
    }
    
    with pytest.raises(AssertionError):
        validate_metrics(metrics)


def test_validate_metrics_invalid_drawdown():
    """Test validate_metrics with invalid drawdown."""
    metrics = {
        'VTE': 2.5,
        'Annualized_Return': 10.0,
        'Annualized_Volatility': 15.0,
        'Sharpe_Ratio': 0.67,
        'Maximum_Drawdown': 150.0,  # > 100%
        'CAGR': 9.5
    }
    
    with pytest.raises(AssertionError):
        validate_metrics(metrics)


def test_validate_metrics_zero_volatility():
    """Test validate_metrics with zero volatility."""
    metrics = {
        'VTE': 2.5,
        'Annualized_Return': 10.0,
        'Annualized_Volatility': 0.0,  # Invalid
        'Sharpe_Ratio': 0.67,
        'Maximum_Drawdown': 25.0,
        'CAGR': 9.5
    }
    
    with pytest.raises(AssertionError):
        validate_metrics(metrics)


def test_vte_different_annualization():
    """Test VTE with different annualization factors."""
    sigma_target_daily = 0.01
    sigma_hat = np.ones(100) * (sigma_target_daily + 0.001)
    
    vte_252 = compute_volatility_tracking_error(sigma_hat, sigma_target_daily, 
                                                annualization_factor=np.sqrt(252))
    vte_365 = compute_volatility_tracking_error(sigma_hat, sigma_target_daily,
                                                annualization_factor=np.sqrt(365))
    
    # Different annualization should give different results
    assert vte_252 != vte_365
    assert vte_365 > vte_252  # sqrt(365) > sqrt(252)


def test_annualized_return_different_periods():
    """Test annualized return with different period counts."""
    returns = np.ones(100) * 0.001
    
    ann_return_252 = compute_annualized_return(returns, periods_per_year=252)
    ann_return_365 = compute_annualized_return(returns, periods_per_year=365)
    
    # Different periods should give different annualized returns
    assert ann_return_365 > ann_return_252


def test_metrics_with_extreme_values():
    """Test metrics computation with extreme values."""
    # Very high returns
    returns = np.ones(100) * 0.1  # 10% daily
    cumulative_returns = np.cumprod(1 + returns)
    sigma_hat = np.ones(100) * 0.05
    
    results = {
        'index_returns': returns,
        'cumulative_returns': cumulative_returns,
        'sigma_hat': sigma_hat
    }
    
    metrics = compute_all_metrics(results, sigma_target_daily=0.01)
    
    # Should still compute without errors
    assert np.all(np.isfinite(list(metrics.values())))
