"""
Monte Carlo confidence band generation for EWMA volatility estimator.

This module implements Experiment 2: generating 10th-90th percentile
confidence bands for the EWMA volatility estimator under known true volatility.
"""

import numpy as np
from typing import Tuple, Dict
from .ewma import compute_ewma_beta


def generate_monte_carlo_confidence_band(
    T: int,
    sigma_true: float,
    halflife: int = 126,
    num_trials: int = 10000,
    periods_per_year: int = 252,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate Monte Carlo confidence band for EWMA volatility estimator.
    
    Simulates M trials of length T with true volatility σ_true,
    computes EWMA vol estimates for each trial, and returns percentiles.
    
    Parameters
    ----------
    T : int
        Number of time periods (trading days)
    sigma_true : float
        True daily volatility
    halflife : int
        EWMA halflife in periods
    num_trials : int
        Number of Monte Carlo trials
    periods_per_year : int
        Number of trading periods per year
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys:
        - 'P10': 10th percentile (length T)
        - 'P50': 50th percentile/median (length T)
        - 'P90': 90th percentile (length T)
        - 'true_vol': True annualized volatility (scalar)
    """
    print(f"Generating Monte Carlo confidence band...")
    print(f"  T = {T} days")
    print(f"  σ_true = {sigma_true:.6f} (daily), {sigma_true * np.sqrt(periods_per_year):.4f} (annual)")
    print(f"  Halflife = {halflife} days")
    print(f"  Num trials = {num_trials}")
    print(f"  Seed = {seed}")
    
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Compute EWMA parameters
    beta = compute_ewma_beta(halflife)
    one_minus_beta = 1.0 - beta
    
    # Precompute normalization factors: 1 - beta^k
    beta_powers = np.power(beta, np.arange(1, T + 1))
    norm_factors = 1.0 - beta_powers
    norm_factors = np.maximum(norm_factors, 1e-10)
    
    # Generate all random returns at once: shape (num_trials, T)
    print("  Generating random returns...")
    returns = rng.normal(0, sigma_true, size=(num_trials, T))
    
    # Compute EWMA volatility for all trials
    print("  Computing EWMA volatility estimates...")
    V = np.zeros((num_trials, T))  # Annualized volatility estimates
    
    # Vectorized computation across trials
    S = np.zeros(num_trials)
    for k in range(T):
        S = one_minus_beta * returns[:, k]**2 + beta * S
        sigma_hat_daily = np.sqrt(S / norm_factors[k])
        V[:, k] = sigma_hat_daily * np.sqrt(periods_per_year)
    
    # Compute percentiles across trials (axis=0)
    print("  Computing percentiles...")
    P10 = np.percentile(V, 10, axis=0)
    P50 = np.percentile(V, 50, axis=0)
    P90 = np.percentile(V, 90, axis=0)
    
    true_vol_annual = sigma_true * np.sqrt(periods_per_year)
    
    print(f"  Done!")
    print(f"  Final band width: [{P10[-1]:.4f}, {P90[-1]:.4f}]")
    print(f"  Final median: {P50[-1]:.4f}")
    print(f"  True vol: {true_vol_annual:.4f}")
    
    return {
        'P10': P10,
        'P50': P50,
        'P90': P90,
        'true_vol': true_vol_annual,
    }


def save_confidence_band(
    band_dict: Dict[str, np.ndarray],
    filepath: str,
    dates: np.ndarray = None
):
    """
    Save confidence band to CSV file.
    
    Parameters
    ----------
    band_dict : Dict[str, np.ndarray]
        Output from generate_monte_carlo_confidence_band
    filepath : str
        Path to save CSV file
    dates : np.ndarray, optional
        Date index (if None, uses integer index)
    """
    import pandas as pd
    
    data = {
        'P10': band_dict['P10'],
        'P50': band_dict['P50'],
        'P90': band_dict['P90'],
    }
    
    if dates is not None:
        df = pd.DataFrame(data, index=dates)
    else:
        df = pd.DataFrame(data)
    
    df.to_csv(filepath)
    print(f"Saved confidence band to {filepath}")


def load_confidence_band(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load confidence band from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with P10, P50, P90 arrays
    """
    import pandas as pd
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    return {
        'P10': df['P10'].values,
        'P50': df['P50'].values,
        'P90': df['P90'].values,
    }


if __name__ == "__main__":
    # Test Monte Carlo confidence band
    print("=== Testing Monte Carlo Confidence Band ===\n")
    
    # Parameters
    T = 252 * 5  # 5 years
    sigma_true = 0.15 / np.sqrt(252)  # 15% annualized
    
    # Generate band
    band = generate_monte_carlo_confidence_band(
        T=T,
        sigma_true=sigma_true,
        halflife=126,
        num_trials=10000,
        seed=42
    )
    
    # Print summary at different time points
    print("\n=== Band Summary at Different Time Points ===")
    for day in [10, 50, 126, 252, T]:
        if day <= T:
            idx = day - 1
            print(f"\nDay {day}:")
            print(f"  P10: {band['P10'][idx]:.4f}")
            print(f"  P50: {band['P50'][idx]:.4f}")
            print(f"  P90: {band['P90'][idx]:.4f}")
            print(f"  Band width: {band['P90'][idx] - band['P10'][idx]:.4f}")
            print(f"  Relative width: {(band['P90'][idx] - band['P10'][idx]) / band['P50'][idx] * 100:.1f}%")
