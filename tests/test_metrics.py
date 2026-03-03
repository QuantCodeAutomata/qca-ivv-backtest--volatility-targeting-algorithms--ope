"""
Tests for performance metrics.
"""

import numpy as np
import pytest
from src.metrics import (
    compute_volatility_tracking_error,
    compute_annualized_return,
    compute_cagr,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_maximum_drawdown,
    compute_all_metrics
)


def test_volatility_tracking_error_zero():
    """Test VTE when estimates match target exactly."""
    sigma_target = 0.01
    sigma_hat = np.ones(100) * sigma_target
    
    vte = compute_volatility_tracking_error(sigma_hat, sigma_target)
    
    assert vte == 0.0


def test_volatility_tracking_error_constant_deviation():
    """Test VTE with constant deviation."""
    sigma_target = 0.01
    sigma_hat = np.ones(100) * 0.012  # 20% higher
    
    vte = compute_volatility_tracking_error(sigma_hat, sigma_target)
    
    # VTE should be |0.012 - 0.01| * sqrt(252) * 100
    expected_vte = 0.002 * np.sqrt(252) * 100
    assert np.abs(vte - expected_vte) < 0.01


def test_volatility_tracking_error_positive():
    """Test that VTE is always non-negative."""
    np.random.seed(42)
    sigma_target = 0.01
    sigma_hat = np.abs(np.random.normal(0.01, 0.002, 100))
    
    vte = compute_volatility_tracking_error(sigma_hat, sigma_target)
    
    assert vte >= 0


def test_annualized_return_zero():
    """Test annualized return with zero returns."""
    returns = np.zeros(252)
    
    ann_ret = compute_annualized_return(returns)
    
    assert ann_ret == 0.0


def test_annualized_return_positive():
    """Test annualized return with positive returns."""
    returns = np.ones(252) * 0.001  # 0.1% daily
    
    ann_ret = compute_annualized_return(returns)
    
    # Should be approximately 0.001 * 252 * 100 = 25.2%
    expected = 0.001 * 252 * 100
    assert np.abs(ann_ret - expected) < 0.01


def test_annualized_return_negative():
    """Test annualized return with negative returns."""
    returns = np.ones(252) * -0.001  # -0.1% daily
    
    ann_ret = compute_annualized_return(returns)
    
    # Should be negative
    assert ann_ret < 0


def test_cagr_no_growth():
    """Test CAGR with no growth."""
    cum_returns = np.ones(252)  # Constant at 1.0
    
    cagr = compute_cagr(cum_returns)
    
    # Should be 0%
    assert np.abs(cagr) < 0.01


def test_cagr_positive_growth():
    """Test CAGR with positive growth."""
    # 10% annual growth over 252 days
    final_level = 1.10
    cum_returns = np.linspace(1.0, final_level, 252)
    
    cagr = compute_cagr(cum_returns)
    
    # Should be close to 10%
    assert 9.5 < cagr < 10.5


def test_cagr_negative_growth():
    """Test CAGR with negative growth."""
    # -10% annual growth
    final_level = 0.90
    cum_returns = np.linspace(1.0, final_level, 252)
    
    cagr = compute_cagr(cum_returns)
    
    # Should be close to -10%
    assert -10.5 < cagr < -9.5


def test_annualized_volatility_zero():
    """Test annualized volatility with zero variance."""
    returns = np.ones(252) * 0.001  # Constant returns
    
    ann_vol = compute_annualized_volatility(returns)
    
    # Should be 0 (no variance)
    assert ann_vol == 0.0


def test_annualized_volatility_known():
    """Test annualized volatility with known volatility."""
    np.random.seed(42)
    daily_vol = 0.01
    returns = np.random.normal(0, daily_vol, 1000)
    
    ann_vol = compute_annualized_volatility(returns)
    
    # Should be approximately daily_vol * sqrt(252) * 100
    expected = daily_vol * np.sqrt(252) * 100
    
    # Allow some sampling error
    assert np.abs(ann_vol - expected) < 2.0


def test_annualized_volatility_positive():
    """Test that annualized volatility is always non-negative."""
    np.random.seed(123)
    returns = np.random.normal(0, 0.01, 252)
    
    ann_vol = compute_annualized_volatility(returns)
    
    assert ann_vol >= 0


def test_sharpe_ratio_zero_vol():
    """Test Sharpe ratio with zero volatility."""
    returns = np.ones(252) * 0.001
    
    sharpe = compute_sharpe_ratio(returns)
    
    # With zero volatility, Sharpe should be 0 (by our convention)
    assert sharpe == 0.0


def test_sharpe_ratio_positive():
    """Test Sharpe ratio with positive returns and volatility."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 252)
    
    sharpe = compute_sharpe_ratio(returns)
    
    # Sharpe should be finite
    assert np.isfinite(sharpe)


def test_sharpe_ratio_calculation():
    """Test Sharpe ratio calculation matches formula."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 252)
    
    sharpe = compute_sharpe_ratio(returns)
    ann_ret = compute_annualized_return(returns)
    ann_vol = compute_annualized_volatility(returns)
    
    # Sharpe = Ann_Ret / Ann_Vol
    expected_sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    assert np.abs(sharpe - expected_sharpe) < 1e-10


def test_maximum_drawdown_no_drawdown():
    """Test max drawdown with monotonically increasing returns."""
    cum_returns = np.linspace(1.0, 2.0, 252)
    
    max_dd = compute_maximum_drawdown(cum_returns)
    
    # Should be 0% (no drawdown)
    assert max_dd == 0.0


def test_maximum_drawdown_simple():
    """Test max drawdown with simple decline."""
    cum_returns = np.array([1.0, 1.1, 1.05, 1.0, 0.9])
    
    max_dd = compute_maximum_drawdown(cum_returns)
    
    # Max was 1.1, min after that was 0.9
    # DD = (1.1 - 0.9) / 1.1 = 0.1818... = 18.18%
    expected = (1.0 - 0.9 / 1.1) * 100
    
    assert np.abs(max_dd - expected) < 0.01


def test_maximum_drawdown_full_loss():
    """Test max drawdown with 50% loss."""
    cum_returns = np.array([1.0, 1.5, 1.0, 0.75])
    
    max_dd = compute_maximum_drawdown(cum_returns)
    
    # Max was 1.5, min after was 0.75
    # DD = (1.5 - 0.75) / 1.5 = 0.5 = 50%
    expected = (1.0 - 0.75 / 1.5) * 100
    
    assert np.abs(max_dd - expected) < 0.01


def test_maximum_drawdown_recovery():
    """Test that recovery after drawdown doesn't change max DD."""
    cum_returns = np.array([1.0, 2.0, 1.0, 2.5])
    
    max_dd = compute_maximum_drawdown(cum_returns)
    
    # Max DD occurred from 2.0 to 1.0 = 50%
    expected = (1.0 - 1.0 / 2.0) * 100
    
    assert np.abs(max_dd - expected) < 0.01


def test_maximum_drawdown_positive():
    """Test that max drawdown is non-negative."""
    np.random.seed(42)
    cum_returns = np.cumprod(1 + np.random.normal(0.0005, 0.01, 252))
    
    max_dd = compute_maximum_drawdown(cum_returns)
    
    assert max_dd >= 0


def test_compute_all_metrics():
    """Test computation of all metrics together."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 252)
    cum_returns = np.cumprod(1 + returns)
    sigma_hat = np.ones(252) * 0.01
    sigma_target = 0.01
    
    metrics = compute_all_metrics(returns, cum_returns, sigma_hat, sigma_target)
    
    # Check all metrics are present
    assert 'VTE' in metrics
    assert 'Ann_Return' in metrics
    assert 'Ann_Volatility' in metrics
    assert 'Sharpe' in metrics
    assert 'Max_Drawdown' in metrics
    assert 'CAGR' in metrics
    assert 'Total_Return' in metrics
    
    # Check all metrics are finite
    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite"


def test_metrics_consistency():
    """Test that metrics are internally consistent."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 252)
    cum_returns = np.cumprod(1 + returns)
    sigma_hat = np.ones(252) * 0.01
    sigma_target = 0.01
    
    metrics = compute_all_metrics(returns, cum_returns, sigma_hat, sigma_target)
    
    # Sharpe = Ann_Return / Ann_Volatility
    expected_sharpe = metrics['Ann_Return'] / metrics['Ann_Volatility']
    assert np.abs(metrics['Sharpe'] - expected_sharpe) < 1e-6
    
    # Total_Return should match final cum_return
    expected_total = (cum_returns[-1] - 1.0) * 100
    assert np.abs(metrics['Total_Return'] - expected_total) < 1e-6


def test_edge_case_single_period():
    """Test metrics with single period."""
    returns = np.array([0.01])
    cum_returns = np.array([1.01])
    sigma_hat = np.array([0.01])
    sigma_target = 0.01
    
    metrics = compute_all_metrics(returns, cum_returns, sigma_hat, sigma_target)
    
    # All metrics should be computable
    assert np.isfinite(metrics['Ann_Return'])
    assert metrics['Max_Drawdown'] == 0.0  # No drawdown with single period


def test_edge_case_all_negative_returns():
    """Test metrics with all negative returns."""
    returns = np.ones(252) * -0.001
    cum_returns = np.cumprod(1 + returns)
    sigma_hat = np.ones(252) * 0.01
    sigma_target = 0.01
    
    metrics = compute_all_metrics(returns, cum_returns, sigma_hat, sigma_target)
    
    # Ann_Return should be negative
    assert metrics['Ann_Return'] < 0
    
    # Max_Drawdown should be large
    assert metrics['Max_Drawdown'] > 0
    
    # Sharpe should be negative
    assert metrics['Sharpe'] < 0
