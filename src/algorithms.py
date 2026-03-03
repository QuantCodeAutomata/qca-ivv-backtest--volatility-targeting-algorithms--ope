"""
Volatility targeting algorithms implementation.

Implements three algorithms:
1. Open-loop inverse-volatility weighting
2. Proportional feedback volatility control
3. Volatility + leverage drawdown control

Plus IVV buy-and-hold baseline.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from src.ewma import compute_ewma_volatility, compute_ewma_ma


@dataclass
class BacktestParameters:
    """Parameters for volatility targeting backtest."""
    
    # Target volatility (daily)
    sigma_target: float = 0.15 / np.sqrt(252)
    
    # Leverage cap
    L: float = 1.5
    
    # EWMA halflife for volatility estimation
    halflife: int = 126
    
    # Volatility controller parameters
    g: float = 50.0  # Controller gain
    kappa_min: float = -1.0
    kappa_max: float = 1.0
    theta: float = 0.5  # Smoothing parameter
    
    # Leverage controller parameters (Algorithm 3)
    g_ell: float = 20.0  # Leverage controller gain
    kappa_lev_min: float = -2.0
    kappa_lev_max: float = 0.0
    h_long: int = 126  # Long MA halflife
    h_short: int = 42   # Short MA halflife
    
    # Control delay (days before engaging controllers)
    control_delay: int = 10


class VolatilityTargetingSimulator:
    """
    Simulator for volatility targeting algorithms.
    
    Maintains all state variables and implements the daily update loop
    for each algorithm variant.
    """
    
    def __init__(self, params: BacktestParameters):
        """
        Initialize simulator.
        
        Parameters
        ----------
        params : BacktestParameters
            Backtest parameters
        """
        self.params = params
        
    def simulate_buy_and_hold(self, risky_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate IVV buy-and-hold strategy.
        
        Parameters
        ----------
        risky_returns : np.ndarray
            Daily risky asset returns
            
        Returns
        -------
        dict
            Dictionary with simulation results
        """
        T = len(risky_returns)
        
        # Cumulative returns
        R = np.zeros(T + 1)
        R[0] = 1.0
        
        for k in range(T):
            R[k + 1] = R[k] * (1 + risky_returns[k])
        
        # Index returns (same as risky returns for buy-and-hold)
        index_returns = risky_returns.copy()
        
        # Running volatility (of risky asset)
        sigma_hat = compute_ewma_volatility(risky_returns, self.params.halflife)
        
        return {
            'index_returns': index_returns,
            'cumulative_returns': R[1:],  # R_k for k=1,...,T
            'sigma_hat': sigma_hat,
            'weights': np.ones(T),  # Always 100% in risky asset
            'cash': np.zeros(T)
        }
    
    def simulate_algorithm_1(self, 
                            risky_returns: np.ndarray,
                            rf_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate Algorithm 1: Open-loop inverse-volatility weighting.
        
        Weight: w_k = min(σ_target / σ_hat_k, L)
        
        Parameters
        ----------
        risky_returns : np.ndarray
            Daily risky asset returns
        rf_returns : np.ndarray
            Daily risk-free returns
            
        Returns
        -------
        dict
            Dictionary with simulation results
        """
        T = len(risky_returns)
        params = self.params
        
        # Initialize state variables
        w = np.zeros(T)
        c = np.zeros(T)
        index_returns = np.zeros(T)
        R_index = np.zeros(T + 1)
        R_index[0] = 1.0
        
        # Initialize weights
        w_prev = 1.0
        c_prev = 0.0
        
        # EWMA volatility (risky asset)
        sigma_hat_risky = compute_ewma_volatility(risky_returns, params.halflife)
        
        for k in range(T):
            # Step 1: Compute index return using previous weights
            r_index = risky_returns[k] * w_prev + rf_returns[k] * c_prev
            index_returns[k] = r_index
            
            # Step 2: Update cumulative index level
            R_index[k + 1] = R_index[k] * (1 + r_index)
            
            # Step 3: Compute new weights (open-loop)
            # Handle edge case: if sigma_hat == 0, use leverage cap
            if sigma_hat_risky[k] > 0:
                w_k = min(params.sigma_target / sigma_hat_risky[k], params.L)
            else:
                w_k = params.L
            
            # Enforce no shorting
            w_k = max(w_k, 0.0)
            c_k = 1.0 - w_k
            
            # Store weights
            w[k] = w_k
            c[k] = c_k
            
            # Update for next iteration
            w_prev = w_k
            c_prev = c_k
        
        # Compute EWMA volatility of index returns
        sigma_hat_index = compute_ewma_volatility(index_returns, params.halflife)
        
        return {
            'index_returns': index_returns,
            'cumulative_returns': R_index[1:],
            'sigma_hat': sigma_hat_index,
            'sigma_hat_risky': sigma_hat_risky,
            'weights': w,
            'cash': c
        }
    
    def simulate_algorithm_2(self,
                            risky_returns: np.ndarray,
                            rf_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate Algorithm 2: Proportional feedback volatility control.
        
        Adds proportional controller to modulate open-loop weight:
        - e_k = log(σ_hat_k^index / σ_target)
        - kappa_k = (1-θ) * clip(-g*e_k, [kappa_min, kappa_max]) + θ * kappa_{k-1}
        - w_k = min(exp(kappa_k) * σ_target / σ_hat_k^risky, L)
        
        Parameters
        ----------
        risky_returns : np.ndarray
            Daily risky asset returns
        rf_returns : np.ndarray
            Daily risk-free returns
            
        Returns
        -------
        dict
            Dictionary with simulation results
        """
        T = len(risky_returns)
        params = self.params
        
        # Initialize state variables
        w = np.zeros(T)
        c = np.zeros(T)
        index_returns = np.zeros(T)
        R_index = np.zeros(T + 1)
        R_index[0] = 1.0
        kappa = np.zeros(T)
        error = np.zeros(T)
        
        # Initialize weights and controller state
        w_prev = 1.0
        c_prev = 0.0
        kappa_prev = 0.0
        
        # EWMA state for recursive computation
        beta = np.exp(-np.log(2) / params.halflife)
        S_risky = 0.0
        S_index = 0.0
        
        for k in range(T):
            # Step 1: Compute index return using previous weights
            r_index = risky_returns[k] * w_prev + rf_returns[k] * c_prev
            index_returns[k] = r_index
            
            # Step 2: Update cumulative index level
            R_index[k + 1] = R_index[k] * (1 + r_index)
            
            # Step 3: Update EWMA volatility (risky asset)
            S_risky = (1 - beta) * risky_returns[k]**2 + beta * S_risky
            norm_factor = 1.0 - beta**(k + 1)
            sigma_hat_risky = np.sqrt(S_risky / norm_factor)
            
            # Step 4: Update EWMA volatility (index)
            S_index = (1 - beta) * r_index**2 + beta * S_index
            sigma_hat_index = np.sqrt(S_index / norm_factor)
            
            # Step 5: Update controller (if past delay period)
            if k < params.control_delay:
                kappa_k = 0.0
                e_k = 0.0
            else:
                # Compute tracking error
                e_k = np.log(sigma_hat_index / params.sigma_target)
                
                # Update kappa with proportional control and smoothing
                kappa_new = -params.g * e_k
                kappa_clipped = np.clip(kappa_new, params.kappa_min, params.kappa_max)
                kappa_k = (1 - params.theta) * kappa_clipped + params.theta * kappa_prev
            
            kappa[k] = kappa_k
            error[k] = e_k if k >= params.control_delay else 0.0
            
            # Step 6: Compute new weights
            if sigma_hat_risky > 0:
                w_k = min(np.exp(kappa_k) * params.sigma_target / sigma_hat_risky, params.L)
            else:
                w_k = params.L
            
            # Enforce no shorting
            w_k = max(w_k, 0.0)
            c_k = 1.0 - w_k
            
            # Store weights
            w[k] = w_k
            c[k] = c_k
            
            # Update for next iteration
            w_prev = w_k
            c_prev = c_k
            kappa_prev = kappa_k
        
        # Compute final sigma_hat for all returns (for metrics)
        sigma_hat_index_full = compute_ewma_volatility(index_returns, params.halflife)
        
        return {
            'index_returns': index_returns,
            'cumulative_returns': R_index[1:],
            'sigma_hat': sigma_hat_index_full,
            'weights': w,
            'cash': c,
            'kappa': kappa,
            'error': error
        }
    
    def simulate_algorithm_3(self,
                            risky_returns: np.ndarray,
                            rf_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate Algorithm 3: Volatility + leverage drawdown control.
        
        Extends Algorithm 2 with dynamic leverage cap:
        - MA_long_k = EWMA(R_index, h_long)
        - MA_short_k = EWMA(R_index, h_short)
        - kappa_lev_k = clip(-g_ell * log(MA_long / MA_short), [kappa_lev_min, 0])
        - effective_L = exp(kappa_lev) * L
        - w_k = min(exp(kappa) * σ_target / σ_hat_risky, effective_L)
        
        Parameters
        ----------
        risky_returns : np.ndarray
            Daily risky asset returns
        rf_returns : np.ndarray
            Daily risk-free returns
            
        Returns
        -------
        dict
            Dictionary with simulation results
        """
        T = len(risky_returns)
        params = self.params
        
        # Initialize state variables
        w = np.zeros(T)
        c = np.zeros(T)
        index_returns = np.zeros(T)
        R_index = np.zeros(T + 1)
        R_index[0] = 1.0
        kappa = np.zeros(T)
        kappa_lev = np.zeros(T)
        error = np.zeros(T)
        
        # Initialize weights and controller state
        w_prev = 1.0
        c_prev = 0.0
        kappa_prev = 0.0
        
        # EWMA state for volatility
        beta = np.exp(-np.log(2) / params.halflife)
        S_risky = 0.0
        S_index = 0.0
        
        # EWMA state for MA (leverage controller)
        beta_long = np.exp(-np.log(2) / params.h_long)
        beta_short = np.exp(-np.log(2) / params.h_short)
        raw_long = 0.0
        raw_short = 0.0
        
        for k in range(T):
            # Step 1: Compute index return using previous weights
            r_index = risky_returns[k] * w_prev + rf_returns[k] * c_prev
            index_returns[k] = r_index
            
            # Step 2: Update cumulative index level
            R_index[k + 1] = R_index[k] * (1 + r_index)
            
            # Step 3: Update EWMA volatility (risky asset)
            S_risky = (1 - beta) * risky_returns[k]**2 + beta * S_risky
            norm_factor = 1.0 - beta**(k + 1)
            sigma_hat_risky = np.sqrt(S_risky / norm_factor)
            
            # Step 4: Update EWMA volatility (index)
            S_index = (1 - beta) * r_index**2 + beta * S_index
            sigma_hat_index = np.sqrt(S_index / norm_factor)
            
            # Step 5a: Update volatility controller
            if k < params.control_delay:
                kappa_k = 0.0
                e_k = 0.0
            else:
                e_k = np.log(sigma_hat_index / params.sigma_target)
                kappa_new = -params.g * e_k
                kappa_clipped = np.clip(kappa_new, params.kappa_min, params.kappa_max)
                kappa_k = (1 - params.theta) * kappa_clipped + params.theta * kappa_prev
            
            kappa[k] = kappa_k
            error[k] = e_k if k >= params.control_delay else 0.0
            
            # Step 5b: Update leverage controller (Algorithm 3 only)
            if k < params.control_delay:
                kappa_lev_k = 0.0
            else:
                # Update EWMA moving averages of cumulative index level
                raw_long = (1 - beta_long) * R_index[k + 1] + beta_long * raw_long
                raw_short = (1 - beta_short) * R_index[k + 1] + beta_short * raw_short
                
                norm_long = 1.0 - beta_long**(k + 1)
                norm_short = 1.0 - beta_short**(k + 1)
                
                MA_long = raw_long / norm_long
                MA_short = raw_short / norm_short
                
                # Compute leverage control signal
                # Handle edge case: if MA_short is zero or negative
                if MA_short > 0 and MA_long > 0:
                    kappa_lev_k = -params.g_ell * np.log(MA_long / MA_short)
                    kappa_lev_k = np.clip(kappa_lev_k, params.kappa_lev_min, params.kappa_lev_max)
                else:
                    kappa_lev_k = 0.0
            
            kappa_lev[k] = kappa_lev_k
            
            # Step 6: Compute new weights with dynamic leverage cap
            effective_L = np.exp(kappa_lev_k) * params.L
            
            if sigma_hat_risky > 0:
                w_k = min(np.exp(kappa_k) * params.sigma_target / sigma_hat_risky, effective_L)
            else:
                w_k = effective_L
            
            # Enforce no shorting
            w_k = max(w_k, 0.0)
            c_k = 1.0 - w_k
            
            # Store weights
            w[k] = w_k
            c[k] = c_k
            
            # Update for next iteration
            w_prev = w_k
            c_prev = c_k
            kappa_prev = kappa_k
        
        # Compute final sigma_hat for all returns
        sigma_hat_index_full = compute_ewma_volatility(index_returns, params.halflife)
        
        return {
            'index_returns': index_returns,
            'cumulative_returns': R_index[1:],
            'sigma_hat': sigma_hat_index_full,
            'weights': w,
            'cash': c,
            'kappa': kappa,
            'kappa_lev': kappa_lev,
            'error': error
        }


def run_all_strategies(risky_returns: np.ndarray,
                       rf_returns: np.ndarray,
                       params: BacktestParameters = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run all four strategies and return results.
    
    Parameters
    ----------
    risky_returns : np.ndarray
        Daily risky asset returns
    rf_returns : np.ndarray
        Daily risk-free returns
    params : BacktestParameters, optional
        Parameters (uses defaults if None)
        
    Returns
    -------
    dict
        Dictionary mapping strategy name to results dictionary
    """
    if params is None:
        params = BacktestParameters()
    
    simulator = VolatilityTargetingSimulator(params)
    
    results = {
        'IVV': simulator.simulate_buy_and_hold(risky_returns),
        'Algorithm_1': simulator.simulate_algorithm_1(risky_returns, rf_returns),
        'Algorithm_2': simulator.simulate_algorithm_2(risky_returns, rf_returns),
        'Algorithm_3': simulator.simulate_algorithm_3(risky_returns, rf_returns)
    }
    
    return results
