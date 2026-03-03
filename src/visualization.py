"""
Visualization module for backtest results.

Creates publication-quality plots:
- Cumulative returns
- Running annualized volatility with Monte Carlo confidence band
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10


def plot_cumulative_returns(
    results_dict: Dict[str, Dict],
    dates: pd.DatetimeIndex,
    save_path: Optional[str] = None,
    use_log_scale: bool = True
):
    """
    Plot cumulative returns for all strategies.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping strategy name to result dictionary
        (must contain 'R_ind' key with cumulative returns)
    dates : pd.DatetimeIndex
        Date index
    save_path : str, optional
        Path to save figure
    use_log_scale : bool
        Whether to use log scale on y-axis
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define colors
    colors = {
        'IVV Buy-and-Hold': '#1f77b4',
        'Algorithm 1 (Open-Loop)': '#ff7f0e',
        'Algorithm 2 (Volatility Control)': '#2ca02c',
        'Algorithm 3 (Leverage Control)': '#d62728',
    }
    
    # Plot each strategy
    for strategy_name, result_data in results_dict.items():
        cumulative = result_data['R_ind']
        color = colors.get(strategy_name, None)
        ax.plot(dates, cumulative, label=strategy_name, linewidth=2, color=color)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (Index Level)', fontsize=12)
    ax.set_title('Cumulative Returns: IVV Buy-and-Hold vs Volatility Targeting Algorithms', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    if use_log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative returns plot to {save_path}")
    
    plt.close()


def plot_running_volatility(
    results_dict: Dict[str, Dict],
    dates: pd.DatetimeIndex,
    mc_band: Optional[Dict[str, np.ndarray]] = None,
    sigma_target_annual: float = 0.15,
    save_path: Optional[str] = None
):
    """
    Plot running annualized volatility with optional MC confidence band.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping strategy name to result dictionary
        (must contain 'sigma_hat_index' key with daily volatility estimates)
    dates : pd.DatetimeIndex
        Date index
    mc_band : Dict[str, np.ndarray], optional
        Monte Carlo confidence band (with keys 'P10', 'P50', 'P90')
    sigma_target_annual : float
        Target annualized volatility (for horizontal line)
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot MC confidence band first (if provided)
    if mc_band is not None:
        ax.fill_between(dates, mc_band['P10'], mc_band['P90'], 
                        color='gray', alpha=0.2, 
                        label='Monte Carlo 10-90% Band (True Vol = 15%)')
    
    # Define colors
    colors = {
        'IVV Buy-and-Hold': '#1f77b4',
        'Algorithm 1 (Open-Loop)': '#ff7f0e',
        'Algorithm 2 (Volatility Control)': '#2ca02c',
        'Algorithm 3 (Leverage Control)': '#d62728',
    }
    
    # Plot each strategy's running volatility
    for strategy_name, result_data in results_dict.items():
        sigma_hat_index = result_data['sigma_hat_index']
        # Annualize
        sigma_hat_annual = sigma_hat_index * np.sqrt(252)
        color = colors.get(strategy_name, None)
        ax.plot(dates, sigma_hat_annual, label=strategy_name, linewidth=1.5, color=color)
    
    # Plot target line
    ax.axhline(y=sigma_target_annual, color='black', linestyle='--', 
               linewidth=2, label=f'Target: {sigma_target_annual*100:.0f}%')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility (EWMA, h=126)', fontsize=12)
    ax.set_title('Running Annualized Volatility: Algorithms vs Buy-and-Hold', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved running volatility plot to {save_path}")
    
    plt.close()


def plot_weights_over_time(
    results_dict: Dict[str, Dict],
    dates: pd.DatetimeIndex,
    save_path: Optional[str] = None
):
    """
    Plot risky asset weights over time for all algorithms.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping strategy name to result dictionary
    dates : pd.DatetimeIndex
        Date index
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = {
        'IVV Buy-and-Hold': '#1f77b4',
        'Algorithm 1 (Open-Loop)': '#ff7f0e',
        'Algorithm 2 (Volatility Control)': '#2ca02c',
        'Algorithm 3 (Leverage Control)': '#d62728',
    }
    
    for strategy_name, result_data in results_dict.items():
        if strategy_name == 'IVV Buy-and-Hold':
            continue  # Skip buy-and-hold (constant weight = 1)
        weights = result_data['w']
        color = colors.get(strategy_name, None)
        ax.plot(dates, weights, label=strategy_name, linewidth=1, alpha=0.8, color=color)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='100% (Unleveraged)')
    ax.axhline(y=1.5, color='red', linestyle=':', linewidth=1, label='150% (Leverage Cap)')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Risky Asset Weight', fontsize=12)
    ax.set_title('Risky Asset Weights Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved weights plot to {save_path}")
    
    plt.close()


def plot_controller_states(
    results_dict: Dict[str, Dict],
    dates: pd.DatetimeIndex,
    save_path: Optional[str] = None
):
    """
    Plot controller states (kappa and kappa_lev) over time.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping strategy name to result dictionary
    dates : pd.DatetimeIndex
        Date index
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot kappa (volatility controller)
    ax1 = axes[0]
    for strategy_name, result_data in results_dict.items():
        if 'kappa' in result_data and np.any(result_data['kappa']):
            ax1.plot(dates, result_data['kappa'], label=strategy_name, linewidth=1, alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=1, color='red', linestyle=':', linewidth=1, label='Upper Bound')
    ax1.axhline(y=-1, color='red', linestyle=':', linewidth=1, label='Lower Bound')
    ax1.set_ylabel('κ (Volatility Controller)', fontsize=12)
    ax1.set_title('Controller States Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot kappa_lev (leverage controller)
    ax2 = axes[1]
    for strategy_name, result_data in results_dict.items():
        if 'kappa_lev' in result_data and np.any(result_data['kappa_lev']):
            ax2.plot(dates, result_data['kappa_lev'], label=strategy_name, linewidth=1, alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=-2, color='red', linestyle=':', linewidth=1, label='Lower Bound')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('κ_ℓ (Leverage Controller)', fontsize=12)
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved controller states plot to {save_path}")
    
    plt.close()


def plot_mc_band_standalone(
    mc_band: Dict[str, np.ndarray],
    T: int,
    sigma_target_annual: float = 0.15,
    save_path: Optional[str] = None
):
    """
    Plot Monte Carlo confidence band standalone (for Experiment 2).
    
    Parameters
    ----------
    mc_band : Dict[str, np.ndarray]
        Monte Carlo confidence band
    T : int
        Number of days
    sigma_target_annual : float
        Target annualized volatility
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    days = np.arange(1, T + 1)
    
    # Plot confidence band
    ax.fill_between(days, mc_band['P10'], mc_band['P90'], 
                    color='lightblue', alpha=0.5, label='10-90% Confidence Band')
    ax.plot(days, mc_band['P50'], color='blue', linewidth=2, label='Median')
    ax.axhline(y=sigma_target_annual, color='red', linestyle='--', 
               linewidth=2, label=f'True Vol: {sigma_target_annual*100:.0f}%')
    
    ax.set_xlabel('Trading Day', fontsize=12)
    ax.set_ylabel('Annualized Volatility Estimate', fontsize=12)
    ax.set_title('Monte Carlo Confidence Band for EWMA Volatility Estimator\n' +
                 f'(True Vol = {sigma_target_annual*100:.0f}%, Halflife = 126 days, 10,000 trials)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved MC band plot to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test visualization with synthetic data
    print("=== Testing Visualization ===\n")
    
    np.random.seed(42)
    T = 252 * 5
    dates = pd.date_range('2020-01-01', periods=T, freq='B')
    
    # Create synthetic results
    results_dict = {}
    for name in ['IVV Buy-and-Hold', 'Algorithm 1 (Open-Loop)', 
                 'Algorithm 2 (Volatility Control)', 'Algorithm 3 (Leverage Control)']:
        cumulative = np.cumprod(1 + np.random.normal(0.0004, 0.01, T))
        sigma_hat = np.ones(T) * 0.15 / np.sqrt(252) + np.random.normal(0, 0.002 / np.sqrt(252), T)
        results_dict[name] = {
            'R_ind': cumulative,
            'sigma_hat_index': sigma_hat,
            'w': np.random.uniform(0.8, 1.5, T),
            'kappa': np.random.uniform(-0.5, 0.5, T),
            'kappa_lev': np.random.uniform(-1, 0, T),
        }
    
    # Test plots
    print("Creating test plots...")
    plot_cumulative_returns(results_dict, dates, '/tmp/test_cumulative.png')
    plot_running_volatility(results_dict, dates, save_path='/tmp/test_volatility.png')
    plot_weights_over_time(results_dict, dates, '/tmp/test_weights.png')
    plot_controller_states(results_dict, dates, '/tmp/test_controllers.png')
    
    print("Test plots created successfully!")
