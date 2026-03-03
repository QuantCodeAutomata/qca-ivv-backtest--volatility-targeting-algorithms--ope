"""
Monte Carlo confidence band generation for EWMA volatility estimator.

Generates 10th-90th percentile confidence bands for the running EWMA
volatility estimator under known true volatility.
"""

import numpy as np
from typing import Tuple, Dict
from src.ewma import compute_ewma_volatility_annualized


def generate_monte_carlo_band(sigma_true_daily: float,
                              T: int,
                              halflife: int = 126,
                              num_trials: int = 10000,
                              percentiles: Tuple[float, float, float] = (10, 50, 90),
                              seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate Monte Carlo confidence band for EWMA volatility estimator.
    
    Simulates M trials of length T from N(0, σ²) and computes the
    percentile envelopes of the running annualized EWMA volatility estimate.
    
    Parameters
    ----------
    sigma_true_daily : float
        True daily volatility (e.g., 0.15/sqrt(252) for 15% annual)
    T : int
        Number of time periods (trading days)
    halflife : int
        EWMA halflife in periods
    num_trials : int
        Number of Monte Carlo trials
    percentiles : tuple
        Percentiles to compute (default: 10th, 50th, 90th)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'P10': 10th percentile band (length T)
        - 'P50': 50th percentile (median) (length T)
        - 'P90': 90th percentile band (length T)
    """
    print(f"Generating Monte Carlo confidence bands...")
    print(f"  True daily volatility: {sigma_true_daily:.6f}")
    print(f"  True annualized volatility: {sigma_true_daily * np.sqrt(252) * 100:.1f}%")
    print(f"  Number of periods: {T}")
    print(f"  Number of trials: {num_trials}")
    print(f"  Halflife: {halflife}")
    print(f"  Random seed: {seed}")
    
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Generate all random returns at once: shape (num_trials, T)
    returns_matrix = rng.normal(0, sigma_true_daily, size=(num_trials, T))
    
    # Compute EWMA volatility for each trial
    volatility_matrix = np.zeros((num_trials, T))
    
    for m in range(num_trials):
        if (m + 1) % 1000 == 0:
            print(f"  Processing trial {m + 1}/{num_trials}...")
        
        vol_ann = compute_ewma_volatility_annualized(returns_matrix[m, :], halflife)
        volatility_matrix[m, :] = vol_ann * 100  # Convert to percent
    
    # Compute percentiles across trials for each time point
    percentile_arrays = np.percentile(volatility_matrix, percentiles, axis=0)
    
    print(f"Monte Carlo simulation complete.")
    print(f"  Median final volatility estimate: {percentile_arrays[1, -1]:.2f}%")
    print(f"  90th percentile final: {percentile_arrays[2, -1]:.2f}%")
    print(f"  10th percentile final: {percentile_arrays[0, -1]:.2f}%")
    
    return {
        'P10': percentile_arrays[0, :],
        'P50': percentile_arrays[1, :],
        'P90': percentile_arrays[2, :]
    }


def compute_band_width(band: Dict[str, np.ndarray], k: int) -> float:
    """
    Compute band width at time k.
    
    Parameters
    ----------
    band : dict
        Band dictionary with P10 and P90
    k : int
        Time index
        
    Returns
    -------
    float
        Band width (P90 - P10) at time k
    """
    return band['P90'][k] - band['P10'][k]


def analyze_band_convergence(band: Dict[str, np.ndarray],
                             sigma_target_annualized_pct: float = 15.0) -> Dict[str, float]:
    """
    Analyze convergence properties of the confidence band.
    
    Parameters
    ----------
    band : dict
        Band dictionary
    sigma_target_annualized_pct : float
        Target annualized volatility in percent
        
    Returns
    -------
    dict
        Analysis results
    """
    T = len(band['P50'])
    
    # Median bias at various time points
    early = int(T * 0.05)  # First 5%
    mid = int(T * 0.5)     # Midpoint
    late = T - 1           # Final
    
    bias_early = band['P50'][early] - sigma_target_annualized_pct
    bias_mid = band['P50'][mid] - sigma_target_annualized_pct
    bias_late = band['P50'][late] - sigma_target_annualized_pct
    
    # Band width at various points
    width_early = compute_band_width(band, early)
    width_mid = compute_band_width(band, mid)
    width_late = compute_band_width(band, late)
    
    return {
        'median_bias_early': bias_early,
        'median_bias_mid': bias_mid,
        'median_bias_late': bias_late,
        'band_width_early': width_early,
        'band_width_mid': width_mid,
        'band_width_late': width_late
    }


def save_band_to_csv(band: Dict[str, np.ndarray],
                     filename: str) -> None:
    """
    Save confidence band to CSV file.
    
    Parameters
    ----------
    band : dict
        Band dictionary
    filename : str
        Output filename
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'P10': band['P10'],
        'P50': band['P50'],
        'P90': band['P90']
    })
    
    df.to_csv(filename, index=False)
    print(f"Saved confidence band to {filename}")
