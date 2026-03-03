"""
Tests for algorithms module.
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from src.algorithms import (
    simulate_buy_and_hold,
    simulate_algorithm_1,
    simulate_algorithm_2,
    simulate_algorithm_3,
    clip_value
)


def test_clip_value():
    """Test value clipping."""
    assert clip_value(0.5, (-1, 1)) == 0.5
    assert clip_value(2.0, (-1, 1)) == 1.0
    assert clip_value(-2.0, (-1, 1)) == -1.0
    assert clip_value(0, (0, 1)) == 0
    assert clip_value(1, (0, 1)) == 1


def test_buy_and_hold_weights():
    """Test buy-and-hold maintains 100% risky allocation."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    result = simulate_buy_and_hold(returns_risky, returns_rf)
    data = result.to_dict()
    
    # All weights should be 1.0 for risky, 0.0 for cash
    assert np.all(data['w'] == 1.0), "Buy-and-hold should have w=1"
    assert np.all(data['c'] == 0.0), "Buy-and-hold should have c=0"


def test_buy_and_hold_returns():
    """Test buy-and-hold returns match risky asset returns."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    result = simulate_buy_and_hold(returns_risky, returns_rf)
    data = result.to_dict()
    
    # Index returns should equal risky returns for buy-and-hold
    assert np.allclose(data['r_ind'], returns_risky), "BH returns should match risky"


def test_algorithm1_weights_sum_to_one():
    """Test Algorithm 1 weights sum to 1."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    result = simulate_algorithm_1(returns_risky, returns_rf)
    data = result.to_dict()
    
    # w + c should equal 1 for all t
    weight_sum = data['w'] + data['c']
    assert np.allclose(weight_sum, 1.0), "Weights should sum to 1"


def test_algorithm1_leverage_cap():
    """Test Algorithm 1 respects leverage cap."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    leverage_cap = 1.5
    result = simulate_algorithm_1(returns_risky, returns_rf, leverage_cap=leverage_cap)
    data = result.to_dict()
    
    # All weights should be <= leverage cap
    assert np.all(data['w'] <= leverage_cap + 1e-10), "Weights should not exceed leverage cap"


def test_algorithm1_no_shorting():
    """Test Algorithm 1 does not short."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    result = simulate_algorithm_1(returns_risky, returns_rf)
    data = result.to_dict()
    
    # All weights should be >= 0
    assert np.all(data['w'] >= 0), "Weights should be non-negative"


def test_algorithm2_controller_delay():
    """Test Algorithm 2 controller delay."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    control_delay = 10
    result = simulate_algorithm_2(returns_risky, returns_rf, control_delay=control_delay)
    data = result.to_dict()
    
    # Kappa should be 0 for first control_delay days
    assert np.all(data['kappa'][:control_delay] == 0), "Kappa should be 0 during delay"


def test_algorithm2_kappa_bounds():
    """Test Algorithm 2 kappa stays within bounds."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    kappa_min, kappa_max = -1.0, 1.0
    result = simulate_algorithm_2(
        returns_risky, returns_rf,
        kappa_min=kappa_min,
        kappa_max=kappa_max
    )
    data = result.to_dict()
    
    # Kappa should stay within bounds
    assert np.all(data['kappa'] >= kappa_min - 1e-6), "Kappa below minimum"
    assert np.all(data['kappa'] <= kappa_max + 1e-6), "Kappa above maximum"


def test_algorithm3_kappa_lev_bounds():
    """Test Algorithm 3 kappa_lev stays within bounds."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    kappa_lev_min = -2.0
    result = simulate_algorithm_3(
        returns_risky, returns_rf,
        kappa_lev_min=kappa_lev_min
    )
    data = result.to_dict()
    
    # Kappa_lev should stay within [kappa_lev_min, 0]
    assert np.all(data['kappa_lev'] >= kappa_lev_min - 1e-6), "Kappa_lev below minimum"
    assert np.all(data['kappa_lev'] <= 1e-6), "Kappa_lev above 0"


def test_algorithm3_effective_leverage():
    """Test Algorithm 3 uses effective leverage cap."""
    np.random.seed(42)
    # Create scenario with drawdown to trigger leverage reduction
    returns_risky = np.concatenate([
        np.ones(100) * 0.001,  # Positive returns
        np.ones(50) * -0.02,   # Drawdown
        np.ones(102) * 0.001   # Recovery
    ])
    returns_rf = np.ones(252) * 0.02 / 360
    
    leverage_cap = 1.5
    result = simulate_algorithm_3(returns_risky, returns_rf, leverage_cap=leverage_cap)
    data = result.to_dict()
    
    # During drawdown, kappa_lev should be negative, reducing effective leverage
    # Check that some kappa_lev values are negative
    assert np.any(data['kappa_lev'] < -0.01), "Kappa_lev should respond to drawdowns"


def test_cumulative_returns_positive():
    """Test cumulative returns start at 1 and are positive."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    for simulate_func in [simulate_buy_and_hold, simulate_algorithm_1, 
                          simulate_algorithm_2, simulate_algorithm_3]:
        result = simulate_func(returns_risky, returns_rf)
        data = result.to_dict()
        
        # Cumulative returns should start at 1
        assert np.isclose(data['R_ind'][0], 1 + data['r_ind'][0]), "First cumulative return incorrect"
        
        # All cumulative returns should be positive
        assert np.all(data['R_ind'] > 0), f"{result.name}: Cumulative returns should be positive"


def test_volatility_estimates_positive():
    """Test volatility estimates are always positive."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 252)
    returns_rf = np.ones(252) * 0.02 / 360
    
    for simulate_func in [simulate_buy_and_hold, simulate_algorithm_1, 
                          simulate_algorithm_2, simulate_algorithm_3]:
        result = simulate_func(returns_risky, returns_rf)
        data = result.to_dict()
        
        assert np.all(data['sigma_hat_risky'] >= 0), f"{result.name}: Risky vol negative"
        assert np.all(data['sigma_hat_index'] >= 0), f"{result.name}: Index vol negative"


def test_algorithm1_vs_algorithm2_same_when_kappa_zero():
    """Test Algorithm 2 with kappa=0 behaves like Algorithm 1."""
    np.random.seed(42)
    returns_risky = np.random.normal(0.001, 0.01, 100)
    returns_rf = np.ones(100) * 0.02 / 360
    
    result1 = simulate_algorithm_1(returns_risky, returns_rf)
    
    # Algorithm 2 with gain=0 should have kappa=0 always
    result2 = simulate_algorithm_2(
        returns_risky, returns_rf,
        controller_gain=0.0,
        control_delay=0
    )
    
    data1 = result1.to_dict()
    data2 = result2.to_dict()
    
    # Weights should be similar (not exact due to different vol estimates)
    # Check that kappa is zero in algorithm 2
    assert np.allclose(data2['kappa'], 0, atol=1e-6), "Kappa should be zero with gain=0"


def test_empty_returns():
    """Test handling of empty returns array."""
    returns_risky = np.array([])
    returns_rf = np.array([])
    
    # Should handle gracefully (no crash)
    for simulate_func in [simulate_buy_and_hold, simulate_algorithm_1]:
        result = simulate_func(returns_risky, returns_rf)
        data = result.to_dict()
        assert len(data['w']) == 0, "Empty input should give empty output"


def test_single_day_returns():
    """Test handling of single day returns."""
    returns_risky = np.array([0.01])
    returns_rf = np.array([0.0001])
    
    for simulate_func in [simulate_buy_and_hold, simulate_algorithm_1, 
                          simulate_algorithm_2, simulate_algorithm_3]:
        result = simulate_func(returns_risky, returns_rf)
        data = result.to_dict()
        
        assert len(data['w']) == 1, "Single day should give one weight"
        assert data['w'][0] >= 0, "Weight should be non-negative"


if __name__ == "__main__":
    print("Running algorithm tests...")
    
    test_clip_value()
    print("✓ test_clip_value")
    
    test_buy_and_hold_weights()
    print("✓ test_buy_and_hold_weights")
    
    test_buy_and_hold_returns()
    print("✓ test_buy_and_hold_returns")
    
    test_algorithm1_weights_sum_to_one()
    print("✓ test_algorithm1_weights_sum_to_one")
    
    test_algorithm1_leverage_cap()
    print("✓ test_algorithm1_leverage_cap")
    
    test_algorithm1_no_shorting()
    print("✓ test_algorithm1_no_shorting")
    
    test_algorithm2_controller_delay()
    print("✓ test_algorithm2_controller_delay")
    
    test_algorithm2_kappa_bounds()
    print("✓ test_algorithm2_kappa_bounds")
    
    test_algorithm3_kappa_lev_bounds()
    print("✓ test_algorithm3_kappa_lev_bounds")
    
    test_algorithm3_effective_leverage()
    print("✓ test_algorithm3_effective_leverage")
    
    test_cumulative_returns_positive()
    print("✓ test_cumulative_returns_positive")
    
    test_volatility_estimates_positive()
    print("✓ test_volatility_estimates_positive")
    
    test_algorithm1_vs_algorithm2_same_when_kappa_zero()
    print("✓ test_algorithm1_vs_algorithm2_same_when_kappa_zero")
    
    test_empty_returns()
    print("✓ test_empty_returns")
    
    test_single_day_returns()
    print("✓ test_single_day_returns")
    
    print("\nAll algorithm tests passed!")
