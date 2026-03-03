"""
EWMA (Exponentially Weighted Moving Average) volatility estimation.

Implements the normalized EWMA estimator with finite-sample correction:
    σ_hat_k^2 = [(1-β)/(1-β^k)] * Σ_{j=1}^k β^{k-j} * x_j^2

where β = exp(-log(2)/h) and h is the halflife in periods.
"""

import numpy as np
from typing import Union


def compute_ewma_beta(halflife: float) -> float:
    """
    Compute EWMA decay parameter β from halflife.
    
    β = exp(-log(2) / halflife)
    
    Parameters
    ----------
    halflife : float
        Halflife in number of periods (e.g., 126 trading days)
        
    Returns
    -------
    float
        Decay parameter β ∈ (0, 1)
    """
    return np.exp(-np.log(2.0) / halflife)


def compute_ewma_volatility_series(returns: np.ndarray, halflife: float = 126.0) -> np.ndarray:
    """
    Compute running EWMA volatility estimates for a return series.
    
    Uses recursive implementation for efficiency:
        S_k = (1-β) * x_k^2 + β * S_{k-1}
        σ_hat_k = sqrt(S_k / (1 - β^k))
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns (length T)
    halflife : float
        EWMA halflife in periods
        
    Returns
    -------
    np.ndarray
        Array of volatility estimates σ_hat_k (length T)
    """
    T = len(returns)
    beta = compute_ewma_beta(halflife)
    one_minus_beta = 1.0 - beta
    
    # Pre-compute normalization factors: 1 / (1 - beta^k) for k=1,...,T
    beta_powers = np.power(beta, np.arange(1, T + 1))
    norm_factors = 1.0 / (1.0 - beta_powers)
    
    # Initialize arrays
    S = np.zeros(T)
    sigma_hat = np.zeros(T)
    
    # Recursive computation
    S[0] = one_minus_beta * returns[0]**2
    sigma_hat[0] = np.sqrt(S[0] * norm_factors[0])
    
    for k in range(1, T):
        S[k] = one_minus_beta * returns[k]**2 + beta * S[k-1]
        sigma_hat[k] = np.sqrt(S[k] * norm_factors[k])
    
    return sigma_hat


def compute_ewma_ma_series(values: np.ndarray, halflife: float) -> np.ndarray:
    """
    Compute running EWMA moving average for a value series.
    
    Uses the same normalized EWMA formula but applied to levels instead of squared returns:
        MA_k = [(1-β)/(1-β^k)] * Σ_{j=1}^k β^{k-j} * v_j
    
    Recursive implementation:
        raw_k = (1-β) * v_k + β * raw_{k-1}
        MA_k = raw_k / (1 - β^k)
    
    Parameters
    ----------
    values : np.ndarray
        Array of values (e.g., cumulative index levels)
    halflife : float
        EWMA halflife in periods
        
    Returns
    -------
    np.ndarray
        Array of EWMA moving averages
    """
    T = len(values)
    beta = compute_ewma_beta(halflife)
    one_minus_beta = 1.0 - beta
    
    # Pre-compute normalization factors
    beta_powers = np.power(beta, np.arange(1, T + 1))
    norm_factors = 1.0 / (1.0 - beta_powers)
    
    # Initialize arrays
    raw = np.zeros(T)
    ma = np.zeros(T)
    
    # Recursive computation
    raw[0] = one_minus_beta * values[0]
    ma[0] = raw[0] * norm_factors[0]
    
    for k in range(1, T):
        raw[k] = one_minus_beta * values[k] + beta * raw[k-1]
        ma[k] = raw[k] * norm_factors[k]
    
    return ma


class EWMAVolatilityEstimator:
    """
    Online EWMA volatility estimator that maintains state.
    
    Useful for step-by-step simulation where we update one observation at a time.
    """
    
    def __init__(self, halflife: float = 126.0):
        """
        Initialize EWMA estimator.
        
        Parameters
        ----------
        halflife : float
            EWMA halflife in periods
        """
        self.halflife = halflife
        self.beta = compute_ewma_beta(halflife)
        self.one_minus_beta = 1.0 - self.beta
        self.reset()
    
    def reset(self) -> None:
        """Reset estimator state."""
        self.k = 0  # Number of observations
        self.S = 0.0  # Unnormalized EWMA sum
    
    def update(self, return_value: float) -> float:
        """
        Update estimator with new return and compute current volatility estimate.
        
        Parameters
        ----------
        return_value : float
            New return observation
            
        Returns
        -------
        float
            Current volatility estimate σ_hat_k
        """
        self.k += 1
        self.S = self.one_minus_beta * return_value**2 + self.beta * self.S
        
        # Normalization factor: 1 / (1 - beta^k)
        norm_factor = 1.0 / (1.0 - self.beta**self.k)
        
        sigma_hat = np.sqrt(self.S * norm_factor)
        return sigma_hat
    
    def get_current_estimate(self) -> float:
        """
        Get current volatility estimate without updating.
        
        Returns
        -------
        float
            Current volatility estimate (0 if no observations yet)
        """
        if self.k == 0:
            return 0.0
        
        norm_factor = 1.0 / (1.0 - self.beta**self.k)
        sigma_hat = np.sqrt(self.S * norm_factor)
        return sigma_hat


class EWMAMovingAverage:
    """
    Online EWMA moving average estimator that maintains state.
    """
    
    def __init__(self, halflife: float):
        """
        Initialize EWMA moving average.
        
        Parameters
        ----------
        halflife : float
            EWMA halflife in periods
        """
        self.halflife = halflife
        self.beta = compute_ewma_beta(halflife)
        self.one_minus_beta = 1.0 - self.beta
        self.reset()
    
    def reset(self) -> None:
        """Reset estimator state."""
        self.k = 0
        self.raw = 0.0
    
    def update(self, value: float) -> float:
        """
        Update with new value and compute current MA.
        
        Parameters
        ----------
        value : float
            New observation
            
        Returns
        -------
        float
            Current moving average estimate
        """
        self.k += 1
        self.raw = self.one_minus_beta * value + self.beta * self.raw
        
        # Normalization factor
        norm_factor = 1.0 / (1.0 - self.beta**self.k)
        
        ma = self.raw * norm_factor
        return ma
    
    def get_current_estimate(self) -> float:
        """
        Get current MA estimate without updating.
        
        Returns
        -------
        float
            Current MA estimate (0 if no observations yet)
        """
        if self.k == 0:
            return 0.0
        
        norm_factor = 1.0 / (1.0 - self.beta**self.k)
        ma = self.raw * norm_factor
        return ma
