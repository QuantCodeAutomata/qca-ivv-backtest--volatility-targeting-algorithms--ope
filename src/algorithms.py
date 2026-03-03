"""
Implementation of volatility targeting algorithms.

This module implements three algorithms from the paper:
1. Algorithm 1: Open-loop inverse-volatility weighting
2. Algorithm 2: Proportional feedback volatility control
3. Algorithm 3: Combined volatility and leverage drawdown control
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .ewma import (
    EWMAVolatilityEstimator,
    EWMAMovingAverage,
    annualize_volatility
)


class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self, name: str):
        self.name = name
        self.weights_risky = []
        self.weights_cash = []
        self.returns_index = []
        self.cumulative_index = []
        self.sigma_hat_risky = []
        self.sigma_hat_index = []
        self.kappa = []
        self.kappa_lev = []
        self.error_signal = []
        self.ma_long = []
        self.ma_short = []
        
    def to_dict(self) -> Dict:
        """Convert to dictionary of arrays."""
        return {
            'w': np.array(self.weights_risky),
            'c': np.array(self.weights_cash),
            'r_ind': np.array(self.returns_index),
            'R_ind': np.array(self.cumulative_index),
            'sigma_hat_risky': np.array(self.sigma_hat_risky),
            'sigma_hat_index': np.array(self.sigma_hat_index),
            'kappa': np.array(self.kappa),
            'kappa_lev': np.array(self.kappa_lev),
            'error': np.array(self.error_signal),
            'ma_long': np.array(self.ma_long),
            'ma_short': np.array(self.ma_short),
        }


def clip_value(value: float, bounds: Tuple[float, float]) -> float:
    """
    Clip value to bounds.
    
    Parameters
    ----------
    value : float
        Value to clip
    bounds : Tuple[float, float]
        (min, max) bounds
        
    Returns
    -------
    float
        Clipped value
    """
    return max(bounds[0], min(value, bounds[1]))


def simulate_buy_and_hold(
    returns_risky: np.ndarray,
    returns_rf: np.ndarray
) -> BacktestResult:
    """
    Simulate IVV buy-and-hold strategy.
    
    This is a 100% allocation to the risky asset (IVV) with no rebalancing.
    
    Parameters
    ----------
    returns_risky : np.ndarray
        Daily simple returns of risky asset
    returns_rf : np.ndarray
        Daily risk-free returns
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    T = len(returns_risky)
    result = BacktestResult("IVV Buy-and-Hold")
    
    # Initialize
    R_cumulative = 1.0
    vol_estimator = EWMAVolatilityEstimator(halflife=126)
    
    for k in range(T):
        # Fixed weights: 100% risky
        w = 1.0
        c = 0.0
        
        # Index return (same as risky return for buy-and-hold)
        r_ind = returns_risky[k]
        
        # Update cumulative return
        R_cumulative *= (1.0 + r_ind)
        
        # Update volatility estimate
        sigma_hat_risky = vol_estimator.update(returns_risky[k])
        sigma_hat_index = sigma_hat_risky  # Same as risky for buy-and-hold
        
        # Store results
        result.weights_risky.append(w)
        result.weights_cash.append(c)
        result.returns_index.append(r_ind)
        result.cumulative_index.append(R_cumulative)
        result.sigma_hat_risky.append(sigma_hat_risky)
        result.sigma_hat_index.append(sigma_hat_index)
        result.kappa.append(0.0)
        result.kappa_lev.append(0.0)
        result.error_signal.append(0.0)
        result.ma_long.append(0.0)
        result.ma_short.append(0.0)
    
    return result


def simulate_algorithm_1(
    returns_risky: np.ndarray,
    returns_rf: np.ndarray,
    sigma_target: float = 0.15 / np.sqrt(252),
    leverage_cap: float = 1.5,
    halflife: int = 126
) -> BacktestResult:
    """
    Simulate Algorithm 1: Open-loop inverse-volatility weighting.
    
    Weight formula: w_k = min(σ_tar / σ̂_k, L)
    
    Parameters
    ----------
    returns_risky : np.ndarray
        Daily simple returns of risky asset
    returns_rf : np.ndarray
        Daily risk-free returns
    sigma_target : float
        Target daily volatility (default: 15% annualized)
    leverage_cap : float
        Maximum leverage (default: 1.5)
    halflife : int
        EWMA halflife (default: 126 days)
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    T = len(returns_risky)
    result = BacktestResult("Algorithm 1 (Open-Loop)")
    
    # Initialize state
    w_prev = 1.0
    c_prev = 0.0
    R_cumulative = 1.0
    
    vol_estimator_risky = EWMAVolatilityEstimator(halflife=halflife)
    vol_estimator_index = EWMAVolatilityEstimator(halflife=halflife)
    
    for k in range(T):
        # Step 1: Compute index return using previous weights
        r_ind = returns_risky[k] * w_prev + returns_rf[k] * c_prev
        
        # Step 2: Update cumulative index level
        R_cumulative *= (1.0 + r_ind)
        
        # Step 3: Update EWMA risky volatility
        sigma_hat_risky = vol_estimator_risky.update(returns_risky[k])
        
        # Step 4: Update EWMA index volatility
        sigma_hat_index = vol_estimator_index.update(r_ind)
        
        # Step 5: Compute new weight (open-loop)
        if sigma_hat_risky > 0:
            w = min(sigma_target / sigma_hat_risky, leverage_cap)
        else:
            # Fallback if sigma is zero (should not happen with real data)
            w = leverage_cap
        
        w = max(w, 0.0)  # No shorting
        c = 1.0 - w
        
        # Validation
        assert abs(w + c - 1.0) < 1e-10, f"Weights don't sum to 1: w={w}, c={c}"
        assert 0 <= w <= leverage_cap + 1e-10, f"Weight out of bounds: w={w}"
        
        # Store results
        result.weights_risky.append(w)
        result.weights_cash.append(c)
        result.returns_index.append(r_ind)
        result.cumulative_index.append(R_cumulative)
        result.sigma_hat_risky.append(sigma_hat_risky)
        result.sigma_hat_index.append(sigma_hat_index)
        result.kappa.append(0.0)
        result.kappa_lev.append(0.0)
        result.error_signal.append(0.0)
        result.ma_long.append(0.0)
        result.ma_short.append(0.0)
        
        # Update for next iteration
        w_prev = w
        c_prev = c
    
    return result


def simulate_algorithm_2(
    returns_risky: np.ndarray,
    returns_rf: np.ndarray,
    sigma_target: float = 0.15 / np.sqrt(252),
    leverage_cap: float = 1.5,
    halflife: int = 126,
    controller_gain: float = 50.0,
    kappa_min: float = -1.0,
    kappa_max: float = 1.0,
    theta: float = 0.5,
    control_delay: int = 10
) -> BacktestResult:
    """
    Simulate Algorithm 2: Proportional feedback volatility control.
    
    Extends Algorithm 1 with feedback controller:
        e_k = log(σ̂^ind_k / σ_tar)
        κ_k = (1-θ) * clip(-g*e_k, [κ_min, κ_max]) + θ*κ_{k-1}
        w_k = min(exp(κ_k) * σ_tar / σ̂_k, L)
    
    Parameters
    ----------
    returns_risky : np.ndarray
        Daily simple returns of risky asset
    returns_rf : np.ndarray
        Daily risk-free returns
    sigma_target : float
        Target daily volatility
    leverage_cap : float
        Maximum leverage
    halflife : int
        EWMA halflife
    controller_gain : float
        Proportional gain g (default: 50)
    kappa_min : float
        Lower bound for κ (default: -1)
    kappa_max : float
        Upper bound for κ (default: 1)
    theta : float
        Smoothing parameter (default: 0.5)
    control_delay : int
        Number of days before engaging controller (default: 10)
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    T = len(returns_risky)
    result = BacktestResult("Algorithm 2 (Volatility Control)")
    
    # Initialize state
    w_prev = 1.0
    c_prev = 0.0
    R_cumulative = 1.0
    kappa_prev = 0.0
    
    vol_estimator_risky = EWMAVolatilityEstimator(halflife=halflife)
    vol_estimator_index = EWMAVolatilityEstimator(halflife=halflife)
    
    for k in range(T):
        # Step 1: Compute index return using previous weights
        r_ind = returns_risky[k] * w_prev + returns_rf[k] * c_prev
        
        # Step 2: Update cumulative index level
        R_cumulative *= (1.0 + r_ind)
        
        # Step 3: Update EWMA risky volatility
        sigma_hat_risky = vol_estimator_risky.update(returns_risky[k])
        
        # Step 4: Update EWMA index volatility
        sigma_hat_index = vol_estimator_index.update(r_ind)
        
        # Step 5: Update controller state
        if k < control_delay:
            # Controller not yet engaged
            kappa = 0.0
            error = 0.0
        else:
            # Compute error signal
            if sigma_hat_index > 0 and sigma_target > 0:
                error = np.log(sigma_hat_index / sigma_target)
            else:
                error = 0.0
            
            # Update kappa with proportional control and smoothing
            kappa_raw = -controller_gain * error
            kappa_clipped = clip_value(kappa_raw, (kappa_min, kappa_max))
            kappa = (1.0 - theta) * kappa_clipped + theta * kappa_prev
        
        # Step 6: Compute new weight with feedback modulation
        if sigma_hat_risky > 0:
            w = min(np.exp(kappa) * sigma_target / sigma_hat_risky, leverage_cap)
        else:
            w = leverage_cap
        
        w = max(w, 0.0)  # No shorting
        c = 1.0 - w
        
        # Validation
        assert abs(w + c - 1.0) < 1e-10, f"Weights don't sum to 1: w={w}, c={c}"
        assert 0 <= w <= leverage_cap + 1e-10, f"Weight out of bounds: w={w}"
        assert kappa_min - 1e-6 <= kappa <= kappa_max + 1e-6, f"Kappa out of bounds: κ={kappa}"
        
        # Store results
        result.weights_risky.append(w)
        result.weights_cash.append(c)
        result.returns_index.append(r_ind)
        result.cumulative_index.append(R_cumulative)
        result.sigma_hat_risky.append(sigma_hat_risky)
        result.sigma_hat_index.append(sigma_hat_index)
        result.kappa.append(kappa)
        result.kappa_lev.append(0.0)
        result.error_signal.append(error if k >= control_delay else 0.0)
        result.ma_long.append(0.0)
        result.ma_short.append(0.0)
        
        # Update for next iteration
        w_prev = w
        c_prev = c
        kappa_prev = kappa
    
    return result


def simulate_algorithm_3(
    returns_risky: np.ndarray,
    returns_rf: np.ndarray,
    sigma_target: float = 0.15 / np.sqrt(252),
    leverage_cap: float = 1.5,
    halflife: int = 126,
    controller_gain: float = 50.0,
    kappa_min: float = -1.0,
    kappa_max: float = 1.0,
    theta: float = 0.5,
    leverage_gain: float = 20.0,
    kappa_lev_min: float = -2.0,
    halflife_long: int = 126,
    halflife_short: int = 42,
    control_delay: int = 10
) -> BacktestResult:
    """
    Simulate Algorithm 3: Combined volatility and leverage drawdown control.
    
    Extends Algorithm 2 with leverage controller:
        κ_ℓ,k = clip(-g_ℓ * log(MA^long_k / MA^short_k), [κ_ℓ,min, 0])
        L_eff = exp(κ_ℓ,k) * L
        w_k = min(exp(κ_k) * σ_tar / σ̂_k, L_eff)
    
    Parameters
    ----------
    returns_risky : np.ndarray
        Daily simple returns of risky asset
    returns_rf : np.ndarray
        Daily risk-free returns
    sigma_target : float
        Target daily volatility
    leverage_cap : float
        Base maximum leverage
    halflife : int
        EWMA halflife for volatility
    controller_gain : float
        Volatility controller gain g
    kappa_min : float
        Lower bound for κ
    kappa_max : float
        Upper bound for κ
    theta : float
        Smoothing parameter for κ
    leverage_gain : float
        Leverage controller gain g_ℓ (default: 20)
    kappa_lev_min : float
        Lower bound for κ_ℓ (default: -2)
    halflife_long : int
        Halflife for long MA (default: 126)
    halflife_short : int
        Halflife for short MA (default: 42)
    control_delay : int
        Number of days before engaging controllers
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    T = len(returns_risky)
    result = BacktestResult("Algorithm 3 (Leverage Control)")
    
    # Initialize state
    w_prev = 1.0
    c_prev = 0.0
    R_cumulative = 1.0
    kappa_prev = 0.0
    
    vol_estimator_risky = EWMAVolatilityEstimator(halflife=halflife)
    vol_estimator_index = EWMAVolatilityEstimator(halflife=halflife)
    ma_long_estimator = EWMAMovingAverage(halflife=halflife_long)
    ma_short_estimator = EWMAMovingAverage(halflife=halflife_short)
    
    for k in range(T):
        # Step 1: Compute index return using previous weights
        r_ind = returns_risky[k] * w_prev + returns_rf[k] * c_prev
        
        # Step 2: Update cumulative index level
        R_cumulative *= (1.0 + r_ind)
        
        # Step 3: Update EWMA risky volatility
        sigma_hat_risky = vol_estimator_risky.update(returns_risky[k])
        
        # Step 4: Update EWMA index volatility
        sigma_hat_index = vol_estimator_index.update(r_ind)
        
        # Update moving averages of cumulative return
        ma_long = ma_long_estimator.update(R_cumulative)
        ma_short = ma_short_estimator.update(R_cumulative)
        
        # Step 5: Update both controllers
        if k < control_delay:
            # Controllers not yet engaged
            kappa = 0.0
            kappa_lev = 0.0
            error = 0.0
        else:
            # Volatility controller
            if sigma_hat_index > 0 and sigma_target > 0:
                error = np.log(sigma_hat_index / sigma_target)
            else:
                error = 0.0
            
            kappa_raw = -controller_gain * error
            kappa_clipped = clip_value(kappa_raw, (kappa_min, kappa_max))
            kappa = (1.0 - theta) * kappa_clipped + theta * kappa_prev
            
            # Leverage controller
            if ma_long > 0 and ma_short > 0:
                ma_ratio_log = np.log(ma_long / ma_short)
                kappa_lev_raw = -leverage_gain * ma_ratio_log
                kappa_lev = clip_value(kappa_lev_raw, (kappa_lev_min, 0.0))
            else:
                kappa_lev = 0.0
        
        # Step 6: Compute new weight with both controllers
        effective_leverage_cap = np.exp(kappa_lev) * leverage_cap
        
        if sigma_hat_risky > 0:
            w = min(np.exp(kappa) * sigma_target / sigma_hat_risky, effective_leverage_cap)
        else:
            w = effective_leverage_cap
        
        w = max(w, 0.0)  # No shorting
        c = 1.0 - w
        
        # Validation
        assert abs(w + c - 1.0) < 1e-10, f"Weights don't sum to 1: w={w}, c={c}"
        # Maximum possible weight is leverage_cap * exp(0) = leverage_cap (when kappa_lev=0)
        assert 0 <= w <= leverage_cap + 1e-6, f"Weight out of bounds: w={w}"
        assert kappa_min - 1e-6 <= kappa <= kappa_max + 1e-6, f"Kappa out of bounds: κ={kappa}"
        assert kappa_lev_min - 1e-6 <= kappa_lev <= 1e-6, f"Kappa_lev out of bounds: κ_ℓ={kappa_lev}"
        
        # Store results
        result.weights_risky.append(w)
        result.weights_cash.append(c)
        result.returns_index.append(r_ind)
        result.cumulative_index.append(R_cumulative)
        result.sigma_hat_risky.append(sigma_hat_risky)
        result.sigma_hat_index.append(sigma_hat_index)
        result.kappa.append(kappa)
        result.kappa_lev.append(kappa_lev)
        result.error_signal.append(error if k >= control_delay else 0.0)
        result.ma_long.append(ma_long)
        result.ma_short.append(ma_short)
        
        # Update for next iteration
        w_prev = w
        c_prev = c
        kappa_prev = kappa
    
    return result


if __name__ == "__main__":
    # Test with synthetic data
    print("=== Testing Algorithms ===\n")
    
    np.random.seed(42)
    T = 252 * 5  # 5 years
    true_vol = 0.15 / np.sqrt(252)
    returns_risky = np.random.normal(0.0003, true_vol, T)  # Small positive drift
    returns_rf = np.ones(T) * 0.02 / 360  # 2% annual risk-free rate
    
    print(f"Simulating {T} days of data...")
    
    # Run all algorithms
    bh_result = simulate_buy_and_hold(returns_risky, returns_rf)
    alg1_result = simulate_algorithm_1(returns_risky, returns_rf)
    alg2_result = simulate_algorithm_2(returns_risky, returns_rf)
    alg3_result = simulate_algorithm_3(returns_risky, returns_rf)
    
    # Print summary
    for result in [bh_result, alg1_result, alg2_result, alg3_result]:
        data = result.to_dict()
        final_wealth = data['R_ind'][-1]
        avg_weight = np.mean(data['w'])
        avg_vol = np.mean(data['sigma_hat_index']) * np.sqrt(252)
        
        print(f"\n{result.name}:")
        print(f"  Final wealth: {final_wealth:.4f}")
        print(f"  Avg weight: {avg_weight:.4f}")
        print(f"  Avg vol (ann): {avg_vol:.2%}")
        if len(data['kappa']) > 0 and np.any(data['kappa']):
            print(f"  Avg κ: {np.mean(data['kappa']):.4f}")
        if len(data['kappa_lev']) > 0 and np.any(data['kappa_lev']):
            print(f"  Avg κ_ℓ: {np.mean(data['kappa_lev']):.4f}")
