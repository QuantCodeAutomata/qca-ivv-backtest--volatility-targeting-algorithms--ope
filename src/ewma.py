"""
EWMA (Exponentially Weighted Moving Average) volatility estimation.

Implements the normalized EWMA volatility estimator with configurable halflife.
Formula: σ_hat_k^2 = [(1-β)/(1-β^k)] * Σ_{j=1}^{k} β^{k-j} * x_j^2
where β = exp(-log(2)/h) for halflife h.

Implemented recursively for computational efficiency.
"""

import numpy as np
from typing import Tuple


def compute_ewma_volatility(returns: np.ndarray, 
                            halflife: int = 126) -> np.ndarray:
    """
    Compute running EWMA volatility estimate (daily, not annualized).
    
    Uses normalized EWMA formula with recursive computation:
    - S_k = (1-β) * r_k^2 + β * S_{k-1}
    - σ_hat_k = sqrt(S_k / (1 - β^k))
    
    Parameters
    ----------
    returns : np.ndarray
        Array of daily returns (length T)
    halflife : int
        Halflife in number of periods (default 126 trading days)
        
    Returns
    -------
    np.ndarray
        Running EWMA volatility estimates (length T)
    """
    T = len(returns)
    
    # Handle empty array edge case
    if T == 0:
        return np.array([])
    
    # Compute decay factor
    beta = np.exp(-np.log(2) / halflife)
    
    # Initialize arrays
    S = np.zeros(T)
    sigma_hat = np.zeros(T)
    
    # Precompute normalization factors: 1 - beta^k
    # For numerical stability with large k, use exact formula
    beta_powers = beta ** np.arange(1, T + 1)
    norm_factors = 1.0 - beta_powers
    
    # Recursive computation
    S[0] = (1 - beta) * returns[0]**2
    sigma_hat[0] = np.sqrt(S[0] / norm_factors[0])
    
    for k in range(1, T):
        S[k] = (1 - beta) * returns[k]**2 + beta * S[k-1]
        sigma_hat[k] = np.sqrt(S[k] / norm_factors[k])
    
    return sigma_hat


def compute_ewma_volatility_annualized(returns: np.ndarray,
                                       halflife: int = 126,
                                       annualization_factor: float = np.sqrt(252)) -> np.ndarray:
    """
    Compute running EWMA volatility estimate (annualized).
    
    Parameters
    ----------
    returns : np.ndarray
        Array of daily returns (length T)
    halflife : int
        Halflife in trading days
    annualization_factor : float
        Factor to annualize daily volatility (default sqrt(252))
        
    Returns
    -------
    np.ndarray
        Running annualized EWMA volatility estimates (length T)
    """
    sigma_hat_daily = compute_ewma_volatility(returns, halflife)
    return sigma_hat_daily * annualization_factor


def compute_ewma_ma(series: np.ndarray,
                    halflife: int) -> np.ndarray:
    """
    Compute running EWMA (moving average) of a level series.
    
    Uses normalized EWMA formula:
    MA_k = [(1-β)/(1-β^k)] * Σ_{j=1}^{k} β^{k-j} * x_j
    
    Implemented recursively:
    - raw_k = (1-β) * x_k + β * raw_{k-1}
    - MA_k = raw_k / (1 - β^k)
    
    Parameters
    ----------
    series : np.ndarray
        Array of values (e.g., cumulative returns)
    halflife : int
        Halflife in number of periods
        
    Returns
    -------
    np.ndarray
        Running EWMA moving average (length T)
    """
    T = len(series)
    
    # Compute decay factor
    beta = np.exp(-np.log(2) / halflife)
    
    # Initialize arrays
    raw = np.zeros(T)
    ma = np.zeros(T)
    
    # Precompute normalization factors
    beta_powers = beta ** np.arange(1, T + 1)
    norm_factors = 1.0 - beta_powers
    
    # Recursive computation
    raw[0] = (1 - beta) * series[0]
    ma[0] = raw[0] / norm_factors[0]
    
    for k in range(1, T):
        raw[k] = (1 - beta) * series[k] + beta * raw[k-1]
        ma[k] = raw[k] / norm_factors[k]
    
    return ma


def validate_ewma_parameters(halflife: int, T: int) -> None:
    """
    Validate EWMA parameters.
    
    Parameters
    ----------
    halflife : int
        Halflife in periods
    T : int
        Length of time series
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if halflife <= 0:
        raise ValueError(f"Halflife must be positive, got {halflife}")
    if T <= 0:
        raise ValueError(f"Time series length must be positive, got {T}")
    if halflife > T:
        print(f"Warning: Halflife ({halflife}) is greater than series length ({T})")


class EWMAVolatilityEstimator:
    """
    Class-based interface for EWMA volatility estimation.
    
    Maintains state for recursive updates.
    """
    
    def __init__(self, halflife: int = 126):
        """
        Initialize EWMA volatility estimator.
        
        Parameters
        ----------
        halflife : int
            Halflife in periods
        """
        self.halflife = halflife
        self.beta = np.exp(-np.log(2) / halflife)
        self.S = 0.0
        self.k = 0
        
    def update(self, return_value: float) -> float:
        """
        Update estimator with new return and compute current volatility.
        
        Parameters
        ----------
        return_value : float
            New return observation
            
        Returns
        -------
        float
            Current volatility estimate (daily, not annualized)
        """
        self.k += 1
        
        # Update sum of squares
        self.S = (1 - self.beta) * return_value**2 + self.beta * self.S
        
        # Compute normalization factor
        norm_factor = 1.0 - self.beta**self.k
        
        # Compute volatility
        sigma_hat = np.sqrt(self.S / norm_factor)
        
        return sigma_hat
    
    def reset(self) -> None:
        """Reset estimator to initial state."""
        self.S = 0.0
        self.k = 0
