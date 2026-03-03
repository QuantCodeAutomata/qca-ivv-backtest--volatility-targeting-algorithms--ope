"""
Monte Carlo simulation for EWMA volatility estimator confidence bands.

Implements Experiment 2: Generate 10th-90th percentile confidence bands
for the running annualized EWMA volatility estimator under known true volatility.
"""

import numpy as np
from typing import Tuple
from src.ewma import compute_ewma_beta


def generate_mc_confidence_band(true_sigma_daily: float,
                                 halflife: float,
                                 n_days: int,
                                 n_trials: int = 10000,
                                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Monte Carlo confidence band for EWMA volatility estimator.
    
    Simulates n_trials of length n_days with true volatility = true_sigma_daily,
    computes EWMA volatility estimates for each trial, and returns percentiles.
    
    Parameters
    ----------
    true_sigma_daily : float
        True daily volatility (e.g., 0.15/sqrt(252) for 15% annualized)
    halflife : float
        EWMA halflife in trading days (e.g., 126)
    n_days : int
        Number of trading days to simulate
    n_trials : int
        Number of Monte Carlo trials
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (P10, P50, P90) percentile bands (annualized volatility)
    """
    print(f"\nGenerating Monte Carlo confidence band...")
    print(f"  True daily vol: {true_sigma_daily:.6f}")
    print(f"  True annual vol: {true_sigma_daily * np.sqrt(252):.4f}")
    print(f"  Halflife: {halflife} days")
    print(f"  Simulation length: {n_days} days")
    print(f"  Number of trials: {n_trials}")
    print(f"  Random seed: {seed}")
    
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Generate all random returns at once: shape (n_trials, n_days)
    returns = rng.normal(0.0, true_sigma_daily, size=(n_trials, n_days))
    
    # Compute EWMA estimates for all trials
    beta = compute_ewma_beta(halflife)
    one_minus_beta = 1.0 - beta
    
    # Pre-compute normalization factors for all k
    k_indices = np.arange(1, n_days + 1)
    beta_powers = np.power(beta, k_indices)
    norm_factors = 1.0 / (1.0 - beta_powers)  # Shape: (n_days,)
    
    # Initialize running EWMA variance estimates
    # V[trial, day] = annualized volatility estimate
    V = np.zeros((n_trials, n_days))
    
    # Compute EWMA recursively across time for all trials
    S = np.zeros(n_trials)  # Unnormalized EWMA sum
    
    for k in range(n_days):
        # Update S for all trials at time k
        S = one_minus_beta * returns[:, k]**2 + beta * S
        
        # Compute volatility estimate
        sigma_hat = np.sqrt(S * norm_factors[k])
        
        # Annualize
        V[:, k] = sigma_hat * np.sqrt(252)
    
    # Compute percentiles across trials (axis=0) for each day
    P10 = np.percentile(V, 10, axis=0)
    P50 = np.percentile(V, 50, axis=0)
    P90 = np.percentile(V, 90, axis=0)
    
    print(f"\nMonte Carlo band statistics:")
    print(f"  Early period (day 10): P10={P10[9]:.2f}%, P50={P50[9]:.2f}%, P90={P90[9]:.2f}%")
    print(f"  Mid period (day {n_days//2}): P10={P10[n_days//2-1]:.2f}%, P50={P50[n_days//2-1]:.2f}%, P90={P90[n_days//2-1]:.2f}%")
    print(f"  Late period (day {n_days}): P10={P10[-1]:.2f}%, P50={P50[-1]:.2f}%, P90={P90[-1]:.2f}%")
    
    return P10, P50, P90


def generate_mc_confidence_band_batch(true_sigma_daily: float,
                                       halflife: float,
                                       n_days: int,
                                       n_trials: int = 10000,
                                       batch_size: int = 1000,
                                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Monte Carlo confidence band with batching for memory efficiency.
    
    Processes trials in batches to avoid memory issues with large n_trials * n_days.
    
    Parameters
    ----------
    true_sigma_daily : float
        True daily volatility
    halflife : float
        EWMA halflife
    n_days : int
        Number of days
    n_trials : int
        Number of trials
    batch_size : int
        Number of trials per batch
    seed : int
        Random seed
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (P10, P50, P90) percentile bands
    """
    print(f"\nGenerating Monte Carlo confidence band (batched)...")
    print(f"  Batch size: {batch_size} trials")
    
    rng = np.random.default_rng(seed)
    beta = compute_ewma_beta(halflife)
    one_minus_beta = 1.0 - beta
    
    # Pre-compute normalization factors
    k_indices = np.arange(1, n_days + 1)
    beta_powers = np.power(beta, k_indices)
    norm_factors = 1.0 / (1.0 - beta_powers)
    
    # Store all volatility estimates across batches
    all_V = []
    
    n_batches = (n_trials + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        # Determine batch size (last batch may be smaller)
        current_batch_size = min(batch_size, n_trials - batch_idx * batch_size)
        
        # Generate returns for this batch
        returns = rng.normal(0.0, true_sigma_daily, size=(current_batch_size, n_days))
        
        # Compute EWMA estimates
        V = np.zeros((current_batch_size, n_days))
        S = np.zeros(current_batch_size)
        
        for k in range(n_days):
            S = one_minus_beta * returns[:, k]**2 + beta * S
            sigma_hat = np.sqrt(S * norm_factors[k])
            V[:, k] = sigma_hat * np.sqrt(252)
        
        all_V.append(V)
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Processed batch {batch_idx + 1}/{n_batches}")
    
    # Concatenate all batches
    all_V = np.vstack(all_V)
    
    # Compute percentiles
    P10 = np.percentile(all_V, 10, axis=0)
    P50 = np.percentile(all_V, 50, axis=0)
    P90 = np.percentile(all_V, 90, axis=0)
    
    print(f"\nMonte Carlo band complete:")
    print(f"  Final P10: {P10[-1]:.2f}%")
    print(f"  Final P50: {P50[-1]:.2f}%")
    print(f"  Final P90: {P90[-1]:.2f}%")
    
    return P10, P50, P90


def save_mc_band(P10: np.ndarray, P50: np.ndarray, P90: np.ndarray, 
                 filepath: str) -> None:
    """
    Save Monte Carlo band to file.
    
    Parameters
    ----------
    P10 : np.ndarray
        10th percentile
    P50 : np.ndarray
        50th percentile (median)
    P90 : np.ndarray
        90th percentile
    filepath : str
        Path to save file
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'P10': P10,
        'P50': P50,
        'P90': P90
    })
    
    df.to_csv(filepath, index=False)
    print(f"Monte Carlo band saved to {filepath}")


def load_mc_band(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Monte Carlo band from file.
    
    Parameters
    ----------
    filepath : str
        Path to load from
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (P10, P50, P90)
    """
    import pandas as pd
    
    df = pd.read_csv(filepath)
    return df['P10'].values, df['P50'].values, df['P90'].values
