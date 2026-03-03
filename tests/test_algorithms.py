"""
Tests for volatility targeting algorithms.

Tests:
- Weight constraints (w >= 0, w + c = 1, w <= L)
- Controller bounds (kappa in [kappa_min, kappa_max])
- Timing (weights use previous period values)
- Return calculation correctness
- Edge cases
"""

import numpy as np
import pytest
from src.algorithms import (
    BacktestParameters,
    VolatilityTargetingSimulator,
    run_all_strategies
)


def test_buy_and_hold_weights():
    """Test that buy-and-hold maintains 100% risky weight."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.random.normal(0.0005, 0.01, 100)
    
    results = simulator.simulate_buy_and_hold(risky_returns)
    
    # All weights should be 1.0
    assert np.all(results['weights'] == 1.0)
    assert np.all(results['cash'] == 0.0)


def test_buy_and_hold_returns():
    """Test that buy-and-hold returns match risky returns."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.random.normal(0.0005, 0.01, 100)
    
    results = simulator.simulate_buy_and_hold(risky_returns)
    
    # Index returns should equal risky returns
    assert np.allclose(results['index_returns'], risky_returns)


def test_algorithm1_weight_constraints():
    """Test that Algorithm 1 satisfies weight constraints."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.random.normal(0.0005, 0.01, 100)
    rf_returns = np.ones(100) * 0.00001
    
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    # Check constraints
    assert np.all(results['weights'] >= 0)  # No shorting
    assert np.all(results['weights'] <= params.L)  # Leverage cap
    assert np.allclose(results['weights'] + results['cash'], 1.0)  # Fully invested


def test_algorithm1_leverage_cap():
    """Test that Algorithm 1 respects leverage cap."""
    params = BacktestParameters(L=1.5)
    simulator = VolatilityTargetingSimulator(params)
    
    # Very low volatility should trigger leverage cap
    risky_returns = np.ones(100) * 0.0001
    rf_returns = np.zeros(100)
    
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    # Weights should hit leverage cap
    assert np.any(results['weights'] >= params.L * 0.99)


def test_algorithm2_kappa_bounds():
    """Test that Algorithm 2 keeps kappa within bounds."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.random.normal(0.0005, 0.01, 200)
    rf_returns = np.ones(200) * 0.00001
    
    results = simulator.simulate_algorithm_2(risky_returns, rf_returns)
    
    # Check kappa bounds (after control delay)
    kappa = results['kappa'][params.control_delay:]
    assert np.all(kappa >= params.kappa_min - 1e-10)
    assert np.all(kappa <= params.kappa_max + 1e-10)


def test_algorithm2_control_delay():
    """Test that Algorithm 2 has zero kappa during control delay."""
    params = BacktestParameters(control_delay=10)
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.random.normal(0.0005, 0.01, 100)
    rf_returns = np.ones(100) * 0.00001
    
    results = simulator.simulate_algorithm_2(risky_returns, rf_returns)
    
    # First control_delay days should have kappa = 0
    assert np.all(results['kappa'][:params.control_delay] == 0)


def test_algorithm3_kappa_lev_bounds():
    """Test that Algorithm 3 keeps kappa_lev within bounds."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.random.normal(0.0005, 0.01, 200)
    rf_returns = np.ones(200) * 0.00001
    
    results = simulator.simulate_algorithm_3(risky_returns, rf_returns)
    
    # Check kappa_lev bounds (after control delay)
    kappa_lev = results['kappa_lev'][params.control_delay:]
    assert np.all(kappa_lev >= params.kappa_lev_min - 1e-10)
    assert np.all(kappa_lev <= params.kappa_lev_max + 1e-10)


def test_algorithm3_weight_constraints():
    """Test that Algorithm 3 satisfies weight constraints."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.random.normal(0.0005, 0.01, 100)
    rf_returns = np.ones(100) * 0.00001
    
    results = simulator.simulate_algorithm_3(risky_returns, rf_returns)
    
    # Check constraints
    assert np.all(results['weights'] >= 0)
    assert np.allclose(results['weights'] + results['cash'], 1.0)


def test_cumulative_returns_monotonic_growth():
    """Test that cumulative returns grow with positive returns."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    # All positive returns
    risky_returns = np.ones(100) * 0.01
    rf_returns = np.ones(100) * 0.0001
    
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    # Cumulative returns should be monotonically increasing
    assert np.all(np.diff(results['cumulative_returns']) > 0)


def test_cumulative_returns_start_at_one():
    """Test that cumulative returns start at 1.0."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.random.normal(0.0005, 0.01, 100)
    rf_returns = np.zeros(100)
    
    for algo_func in [simulator.simulate_algorithm_1,
                     simulator.simulate_algorithm_2,
                     simulator.simulate_algorithm_3]:
        results = algo_func(risky_returns, rf_returns)
        
        # First cumulative return should be close to 1 + first index return
        expected_first = 1.0 * (1 + results['index_returns'][0])
        assert np.isclose(results['cumulative_returns'][0], expected_first)


def test_index_return_calculation():
    """Test that index returns are calculated correctly."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.array([0.01, -0.02, 0.015])
    rf_returns = np.array([0.0001, 0.0001, 0.0001])
    
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    # Manual verification for first return
    # r_1^ind = r_1 * w_0 + r_rf_1 * c_0
    # w_0 = 1, c_0 = 0 (initialization)
    expected_first = risky_returns[0] * 1.0 + rf_returns[0] * 0.0
    assert np.isclose(results['index_returns'][0], expected_first)


def test_zero_volatility_handling():
    """Test handling of zero volatility edge case."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    # All zero returns (zero volatility)
    risky_returns = np.zeros(50)
    rf_returns = np.zeros(50)
    
    # Should not crash and should use leverage cap as fallback
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    # Should have used leverage cap (or similar safe value)
    assert np.all(np.isfinite(results['weights']))
    assert np.all(results['weights'] >= 0)


def test_high_volatility_reduces_weight():
    """Test that high volatility reduces risky weight."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    # High volatility period
    risky_returns = np.random.normal(0, 0.05, 200)  # High vol
    rf_returns = np.zeros(200)
    
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    # Average weight should be below 1.0 due to high vol
    avg_weight = np.mean(results['weights'][50:])  # Skip initial period
    assert avg_weight < 1.0


def test_low_volatility_increases_weight():
    """Test that low volatility increases risky weight (up to leverage cap)."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    # Low volatility period
    risky_returns = np.random.normal(0, 0.002, 200)  # Low vol
    rf_returns = np.zeros(200)
    
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    # Some weights should be above 1.0 (leveraged)
    assert np.any(results['weights'] > 1.0)


def test_run_all_strategies():
    """Test that run_all_strategies executes all four strategies."""
    risky_returns = np.random.normal(0.0005, 0.01, 100)
    rf_returns = np.ones(100) * 0.00001
    
    all_results = run_all_strategies(risky_returns, rf_returns)
    
    # Should have all four strategies
    assert 'IVV' in all_results
    assert 'Algorithm_1' in all_results
    assert 'Algorithm_2' in all_results
    assert 'Algorithm_3' in all_results
    
    # Each should have required fields
    for name, results in all_results.items():
        assert 'index_returns' in results
        assert 'cumulative_returns' in results
        assert 'sigma_hat' in results
        assert len(results['index_returns']) == 100


def test_negative_returns_handling():
    """Test that algorithms handle negative returns correctly."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    # Large negative return (crash scenario)
    risky_returns = np.array([0.01] * 50 + [-0.3] + [0.01] * 49)
    rf_returns = np.zeros(100)
    
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    # Should handle without errors
    assert np.all(np.isfinite(results['cumulative_returns']))
    assert np.all(results['cumulative_returns'] > 0)  # Should stay positive


def test_algorithm2_smoothing_effect():
    """Test that theta parameter in Algorithm 2 smooths kappa."""
    params_smooth = BacktestParameters(theta=0.9)  # High smoothing
    params_nosmooth = BacktestParameters(theta=0.0)  # No smoothing
    
    risky_returns = np.random.normal(0.0005, 0.01, 200)
    rf_returns = np.ones(200) * 0.00001
    
    sim_smooth = VolatilityTargetingSimulator(params_smooth)
    sim_nosmooth = VolatilityTargetingSimulator(params_nosmooth)
    
    results_smooth = sim_smooth.simulate_algorithm_2(risky_returns, rf_returns)
    results_nosmooth = sim_nosmooth.simulate_algorithm_2(risky_returns, rf_returns)
    
    # Smoothed kappa should have lower variance
    kappa_smooth_var = np.var(results_smooth['kappa'][20:])
    kappa_nosmooth_var = np.var(results_nosmooth['kappa'][20:])
    
    assert kappa_smooth_var < kappa_nosmooth_var


def test_empty_returns():
    """Test handling of empty returns array."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.array([])
    rf_returns = np.array([])
    
    # Should handle gracefully (may return empty arrays)
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    assert len(results['index_returns']) == 0
    assert len(results['cumulative_returns']) == 0


def test_single_return():
    """Test algorithms with single return observation."""
    params = BacktestParameters()
    simulator = VolatilityTargetingSimulator(params)
    
    risky_returns = np.array([0.01])
    rf_returns = np.array([0.0001])
    
    results = simulator.simulate_algorithm_1(risky_returns, rf_returns)
    
    assert len(results['index_returns']) == 1
    assert len(results['cumulative_returns']) == 1
    assert np.isfinite(results['index_returns'][0])
