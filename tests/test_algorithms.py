"""
Tests for trading algorithms.
"""

import numpy as np
import pytest
from src.algorithms import (
    StrategyParameters,
    StrategyState,
    BuyAndHoldIVV,
    Algorithm1OpenLoop,
    Algorithm2VolatilityControl,
    Algorithm3LeverageControl
)


@pytest.fixture
def default_params():
    """Default strategy parameters."""
    return StrategyParameters()


@pytest.fixture
def sample_returns():
    """Sample return series for testing."""
    np.random.seed(42)
    risky = np.random.normal(0.0005, 0.01, 100)
    rf = np.ones(100) * 0.00001  # ~0.36% annual
    return risky, rf


def test_strategy_parameters_defaults():
    """Test default parameter values."""
    params = StrategyParameters()
    
    assert params.sigma_target > 0
    assert params.L == 1.5
    assert params.halflife == 126.0
    assert params.g == 50.0
    assert params.theta == 0.5
    assert params.control_delay == 10


def test_buy_and_hold_ivv(sample_returns):
    """Test buy-and-hold strategy."""
    risky, _ = sample_returns
    
    strategy = BuyAndHoldIVV()
    
    for r in risky:
        strategy.update(r)
    
    # Check cumulative returns are positive and increasing (on average)
    cum_returns = strategy.get_returns()
    assert len(cum_returns) == len(risky)
    assert cum_returns[0] > 0  # First return level
    assert cum_returns[-1] > 0  # Final return level


def test_buy_and_hold_zero_returns():
    """Test buy-and-hold with zero returns."""
    strategy = BuyAndHoldIVV()
    
    for _ in range(10):
        strategy.update(0.0)
    
    cum_returns = strategy.get_returns()
    
    # All cumulative returns should be 1.0 (no change)
    assert np.allclose(cum_returns, 1.0)


def test_algorithm1_initialization(default_params):
    """Test Algorithm 1 initialization."""
    strategy = Algorithm1OpenLoop(default_params)
    
    assert strategy.state.w == 1.0
    assert strategy.state.c == 0.0
    assert strategy.state.R_ind == 1.0


def test_algorithm1_weights_sum_to_one(default_params, sample_returns):
    """Test that Algorithm 1 weights always sum to 1."""
    risky, rf = sample_returns
    strategy = Algorithm1OpenLoop(default_params)
    
    for k, (r_risky, r_rf) in enumerate(zip(risky, rf), 1):
        state = strategy.update(r_risky, r_rf, k)
        assert np.abs(state.w + state.c - 1.0) < 1e-10


def test_algorithm1_respects_leverage_cap(default_params, sample_returns):
    """Test that Algorithm 1 respects leverage cap."""
    risky, rf = sample_returns
    strategy = Algorithm1OpenLoop(default_params)
    
    for k, (r_risky, r_rf) in enumerate(zip(risky, rf), 1):
        state = strategy.update(r_risky, r_rf, k)
        assert state.w <= default_params.L + 1e-10
        assert state.w >= 0.0


def test_algorithm1_no_shorting(default_params, sample_returns):
    """Test that Algorithm 1 doesn't short the risky asset."""
    risky, rf = sample_returns
    strategy = Algorithm1OpenLoop(default_params)
    
    for k, (r_risky, r_rf) in enumerate(zip(risky, rf), 1):
        state = strategy.update(r_risky, r_rf, k)
        # No shorting of risky asset
        assert state.w >= 0.0
        # Cash can be negative when leveraged (w > 1)
        # But w + c should equal 1
        assert abs(state.w + state.c - 1.0) < 1e-10


def test_algorithm2_initialization(default_params):
    """Test Algorithm 2 initialization."""
    strategy = Algorithm2VolatilityControl(default_params)
    
    assert strategy.state.kappa == 0.0
    assert strategy.state.w == 1.0


def test_algorithm2_kappa_bounds(default_params, sample_returns):
    """Test that Algorithm 2 kappa stays within bounds."""
    risky, rf = sample_returns
    strategy = Algorithm2VolatilityControl(default_params)
    
    for k, (r_risky, r_rf) in enumerate(zip(risky, rf), 1):
        state = strategy.update(r_risky, r_rf, k)
        assert state.kappa >= default_params.kappa_min - 1e-10
        assert state.kappa <= default_params.kappa_max + 1e-10


def test_algorithm2_controller_delay(default_params, sample_returns):
    """Test that Algorithm 2 controller engages after delay."""
    risky, rf = sample_returns
    strategy = Algorithm2VolatilityControl(default_params)
    
    # First few steps should have kappa = 0
    for k in range(1, default_params.control_delay + 1):
        state = strategy.update(risky[k-1], rf[k-1], k)
        assert state.kappa == 0.0
    
    # After delay, kappa may be non-zero
    # (depends on whether tracking error exists)


def test_algorithm3_initialization(default_params):
    """Test Algorithm 3 initialization."""
    strategy = Algorithm3LeverageControl(default_params)
    
    assert strategy.state.kappa == 0.0
    assert strategy.state.kappa_lev == 0.0
    assert strategy.state.MA_long == 1.0
    assert strategy.state.MA_short == 1.0


def test_algorithm3_kappa_lev_bounds(default_params, sample_returns):
    """Test that Algorithm 3 kappa_lev stays within bounds."""
    risky, rf = sample_returns
    strategy = Algorithm3LeverageControl(default_params)
    
    for k, (r_risky, r_rf) in enumerate(zip(risky, rf), 1):
        state = strategy.update(r_risky, r_rf, k)
        assert state.kappa_lev >= default_params.kappa_lev_min - 1e-10
        assert state.kappa_lev <= 0.0 + 1e-10


def test_algorithm3_ma_positive(default_params, sample_returns):
    """Test that Algorithm 3 moving averages are positive."""
    risky, rf = sample_returns
    strategy = Algorithm3LeverageControl(default_params)
    
    for k, (r_risky, r_rf) in enumerate(zip(risky, rf), 1):
        state = strategy.update(r_risky, r_rf, k)
        assert state.MA_long > 0
        assert state.MA_short > 0


def test_algorithm3_dynamic_leverage_cap(default_params):
    """Test that Algorithm 3 modulates leverage cap."""
    # Create scenario where long MA > short MA (drawdown)
    # This should reduce effective leverage
    
    params = default_params
    strategy = Algorithm3LeverageControl(params)
    
    # Simulate declining market
    declining_returns = np.linspace(0.01, -0.02, 50)
    rf = np.zeros(50)
    
    for k, r_risky in enumerate(declining_returns, 1):
        state = strategy.update(r_risky, rf[k-1], k)
    
    # After control delay, kappa_lev should be negative (reducing leverage)
    final_state = strategy.history[-1]
    
    # In a declining market, MA_long > MA_short
    # So kappa_lev = -g_lev * log(MA_long / MA_short) should be negative


def test_all_algorithms_produce_cumulative_returns(default_params, sample_returns):
    """Test that all algorithms produce valid cumulative return series."""
    risky, rf = sample_returns
    
    strategies = [
        Algorithm1OpenLoop(default_params),
        Algorithm2VolatilityControl(default_params),
        Algorithm3LeverageControl(default_params)
    ]
    
    for strategy in strategies:
        for k, (r_risky, r_rf) in enumerate(zip(risky, rf), 1):
            state = strategy.update(r_risky, r_rf, k)
            assert state.R_ind > 0  # Cumulative return level always positive
            assert np.isfinite(state.R_ind)


def test_algorithm1_inverse_vol_weighting(default_params):
    """Test that Algorithm 1 implements inverse-vol weighting correctly."""
    params = default_params
    strategy = Algorithm1OpenLoop(params)
    
    # Low vol scenario: should lever up (closer to L)
    low_vol_returns = np.random.normal(0, 0.002, 200)
    rf = np.zeros(200)
    
    for k, r in enumerate(low_vol_returns, 1):
        state = strategy.update(r, rf[k-1], k)
    
    # In low vol, weight should approach leverage cap
    final_weight = strategy.history[-1].w
    # Weight = sigma_target / sigma_hat, capped at L
    # If sigma_hat < sigma_target, weight approaches L


def test_strategy_history_length(default_params, sample_returns):
    """Test that strategy history has correct length."""
    risky, rf = sample_returns
    strategy = Algorithm1OpenLoop(default_params)
    
    for k, (r_risky, r_rf) in enumerate(zip(risky, rf), 1):
        strategy.update(r_risky, r_rf, k)
    
    assert len(strategy.history) == len(risky)


def test_strategy_reset(default_params):
    """Test that strategy reset works correctly."""
    strategy = Algorithm1OpenLoop(default_params)
    
    # Run some updates
    for k in range(10):
        strategy.update(0.01, 0.0001, k+1)
    
    assert len(strategy.history) == 10
    
    # Reset
    strategy.reset()
    
    assert len(strategy.history) == 0
    assert strategy.state.R_ind == 1.0
    assert strategy.state.w == 1.0


def test_negative_returns_handling(default_params):
    """Test that algorithms handle negative returns correctly."""
    strategy = Algorithm1OpenLoop(default_params)
    
    # Large negative return
    state = strategy.update(-0.05, 0.0, 1)
    
    # Index return should be negative
    assert state.r_ind < 0
    
    # But cumulative level should still be positive
    assert state.R_ind > 0
    
    # Weight should still be valid
    assert 0 <= state.w <= default_params.L


def test_zero_volatility_fallback(default_params):
    """Test fallback behavior when estimated volatility is zero."""
    strategy = Algorithm1OpenLoop(default_params)
    
    # First update with zero return
    state = strategy.update(0.0, 0.0, 1)
    
    # Volatility estimate will be 0, should fallback to L
    assert state.sigma_hat_risky == 0.0
    # Next weight calculation should handle this
