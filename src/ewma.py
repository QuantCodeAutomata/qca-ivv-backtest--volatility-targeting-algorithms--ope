"""
EWMA (Exponentially Weighted Moving Average) volatility and moving average estimators.

This module implements the normalized EWMA estimators used in the paper:
- EWMA volatility estimator with specified halflife
- EWMA moving average for cumulative returns
"""

import numpy as np
from typing import Tuple


def compute_ewma_beta(halflife: int) -> float:
    """
    Compute EWMA decay parameter beta from halflife.
    
    Parameters
    ----------
    halflife : int
        Halflife in number of periods (trading days)
        
    Returns
    -------
    float
        Beta decay parameter: β = exp(-log(2) / halflife)
    """
    return np.exp(-np.log(2.0) / halflife)


def ewma_volatility_recursive(
    returns: np.ndarray,
    halflife: int = 126
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute running EWMA volatility estimates using recursive formula.
    
    The normalized EWMA variance estimator:
        σ̂²_k = [(1-β)/(1-β^k)] * Σ_{j=1}^k β^(k-j) * x_j²
    
    Implemented recursively:
        S_k = (1-β) * x_k² + β * S_{k-1}
        σ̂_k = √(S_k / (1-β^k))
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns (length T)
    halflife : int
        EWMA halflife in periods (default: 126 trading days)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - sigma_hat: Running volatility estimates (length T)
        - S: Running sum state (for debugging/inspection)
    """
    T = len(returns)
    beta = compute_ewma_beta(halflife)
    one_minus_beta = 1.0 - beta
    
    # Initialize arrays
    S = np.zeros(T)
    sigma_hat = np.zeros(T)
    
    # Precompute normalization factors: 1 - beta^k
    # For numerical stability with large k, cap at 1.0
    beta_powers = np.power(beta, np.arange(1, T + 1))
    norm_factors = 1.0 - beta_powers
    norm_factors = np.maximum(norm_factors, 1e-10)  # Avoid division by zero
    
    # Recursive computation
    S_prev = 0.0
    for k in range(T):
        S[k] = one_minus_beta * returns[k]**2 + beta * S_prev
        sigma_hat[k] = np.sqrt(S[k] / norm_factors[k])
        S_prev = S[k]
    
    return sigma_hat, S


def ewma_moving_average_recursive(
    levels: np.ndarray,
    halflife: int
) -> np.ndarray:
    """
    Compute running EWMA moving average of a level series.
    
    The normalized EWMA:
        MA_k = [(1-β)/(1-β^k)] * Σ_{j=1}^k β^(k-j) * R_j
    
    Implemented recursively:
        raw_k = (1-β) * R_k + β * raw_{k-1}
        MA_k = raw_k / (1-β^k)
    
    Parameters
    ----------
    levels : np.ndarray
        Array of level values (e.g., cumulative returns)
    halflife : int
        EWMA halflife in periods
        
    Returns
    -------
    np.ndarray
        Running moving average estimates
    """
    T = len(levels)
    beta = compute_ewma_beta(halflife)
    one_minus_beta = 1.0 - beta
    
    # Initialize arrays
    raw = np.zeros(T)
    MA = np.zeros(T)
    
    # Precompute normalization factors
    beta_powers = np.power(beta, np.arange(1, T + 1))
    norm_factors = 1.0 - beta_powers
    norm_factors = np.maximum(norm_factors, 1e-10)
    
    # Recursive computation
    raw_prev = 0.0
    for k in range(T):
        raw[k] = one_minus_beta * levels[k] + beta * raw_prev
        MA[k] = raw[k] / norm_factors[k]
        raw_prev = raw[k]
    
    return MA


def annualize_volatility(daily_vol: np.ndarray, periods_per_year: int = 252) -> np.ndarray:
    """
    Convert daily volatility to annualized volatility.
    
    Parameters
    ----------
    daily_vol : np.ndarray
        Daily volatility estimates
    periods_per_year : int
        Number of trading periods per year (default: 252)
        
    Returns
    -------
    np.ndarray
        Annualized volatility
    """
    return daily_vol * np.sqrt(periods_per_year)


def compute_target_daily_volatility(
    annual_target: float = 0.15,
    periods_per_year: int = 252
) -> float:
    """
    Compute target daily volatility from annual target.
    
    Parameters
    ----------
    annual_target : float
        Target annualized volatility (e.g., 0.15 for 15%)
    periods_per_year : int
        Number of trading periods per year (default: 252)
        
    Returns
    -------
    float
        Target daily volatility
    """
    return annual_target / np.sqrt(periods_per_year)


class EWMAVolatilityEstimator:
    """
    Class-based EWMA volatility estimator for online/streaming updates.
    
    This is useful for simulations where we need to maintain state.
    """
    
    def __init__(self, halflife: int = 126):
        """
        Initialize EWMA estimator.
        
        Parameters
        ----------
        halflife : int
            EWMA halflife in periods
        """
        self.halflife = halflife
        self.beta = compute_ewma_beta(halflife)
        self.one_minus_beta = 1.0 - self.beta
        
        # State variables
        self.S = 0.0
        self.k = 0
        
    def update(self, return_value: float) -> float:
        """
        Update estimator with new return and compute volatility.
        
        Parameters
        ----------
        return_value : float
            New return observation
            
        Returns
        -------
        float
            Updated volatility estimate
        """
        self.k += 1
        self.S = self.one_minus_beta * return_value**2 + self.beta * self.S
        
        # Normalization factor
        beta_power_k = self.beta ** self.k
        norm_factor = 1.0 - beta_power_k
        norm_factor = max(norm_factor, 1e-10)
        
        sigma_hat = np.sqrt(self.S / norm_factor)
        return sigma_hat
    
    def reset(self):
        """Reset estimator state."""
        self.S = 0.0
        self.k = 0


class EWMAMovingAverage:
    """
    Class-based EWMA moving average for online/streaming updates.
    """
    
    def __init__(self, halflife: int):
        """
        Initialize EWMA moving average.
        
        Parameters
        ----------
        halflife : int
            EWMA halflife in periods
        """
        self.halflife = halflife
        self.beta = compute_ewma_beta(halflife)
        self.one_minus_beta = 1.0 - self.beta
        
        # State variables
        self.raw = 0.0
        self.k = 0
        
    def update(self, level: float) -> float:
        """
        Update MA with new level value.
        
        Parameters
        ----------
        level : float
            New level observation
            
        Returns
        -------
        float
            Updated MA estimate
        """
        self.k += 1
        self.raw = self.one_minus_beta * level + self.beta * self.raw
        
        # Normalization factor
        beta_power_k = self.beta ** self.k
        norm_factor = 1.0 - beta_power_k
        norm_factor = max(norm_factor, 1e-10)
        
        ma = self.raw / norm_factor
        return ma
    
    def reset(self):
        """Reset MA state."""
        self.raw = 0.0
        self.k = 0


if __name__ == "__main__":
    # Test EWMA functions
    print("=== Testing EWMA Functions ===\n")
    
    # Generate synthetic returns
    np.random.seed(42)
    T = 1000
    true_vol = 0.15 / np.sqrt(252)
    returns = np.random.normal(0, true_vol, T)
    
    # Compute EWMA volatility
    sigma_hat, S = ewma_volatility_recursive(returns, halflife=126)
    sigma_hat_ann = annualize_volatility(sigma_hat)
    
    print(f"Synthetic returns (N={T}):")
    print(f"  True daily vol: {true_vol:.6f}")
    print(f"  True annual vol: {true_vol * np.sqrt(252):.4f}")
    print(f"  Sample std: {returns.std():.6f}")
    print(f"  Sample std (ann): {returns.std() * np.sqrt(252):.4f}")
    print(f"\nEWMA estimates (halflife=126):")
    print(f"  Final daily vol: {sigma_hat[-1]:.6f}")
    print(f"  Final annual vol: {sigma_hat_ann[-1]:.4f}")
    print(f"  Mean annual vol: {sigma_hat_ann.mean():.4f}")
    
    # Test online estimator
    estimator = EWMAVolatilityEstimator(halflife=126)
    online_vols = []
    for r in returns:
        vol = estimator.update(r)
        online_vols.append(vol)
    online_vols = np.array(online_vols)
    
    print(f"\nOnline estimator:")
    print(f"  Final daily vol: {online_vols[-1]:.6f}")
    print(f"  Match batch: {np.allclose(online_vols, sigma_hat)}")
    
    # Test MA
    levels = np.cumsum(returns) + 1.0
    ma_long = ewma_moving_average_recursive(levels, halflife=126)
    ma_short = ewma_moving_average_recursive(levels, halflife=42)
    
    print(f"\nEWMA Moving Averages:")
    print(f"  Final level: {levels[-1]:.6f}")
    print(f"  MA long (h=126): {ma_long[-1]:.6f}")
    print(f"  MA short (h=42): {ma_short[-1]:.6f}")
