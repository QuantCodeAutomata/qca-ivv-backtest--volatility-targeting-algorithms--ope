"""
Visualization module for plotting results.

Creates:
- Cumulative returns plot
- Running volatility plot with Monte Carlo confidence bands
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


def plot_cumulative_returns(dates: pd.DatetimeIndex,
                            cumulative_returns: Dict[str, np.ndarray],
                            save_path: Optional[str] = None) -> None:
    """
    Plot cumulative returns for all strategies.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading dates
    cumulative_returns : Dict[str, np.ndarray]
        Dictionary mapping strategy names to cumulative return arrays
    save_path : Optional[str]
        Path to save figure (if None, display only)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors and line styles for each strategy
    styles = {
        'IVV Buy-and-Hold': {'color': 'gray', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.7},
        'Algorithm 1 (Open-Loop)': {'color': 'blue', 'linestyle': '--', 'linewidth': 2},
        'Algorithm 2 (Vol Control)': {'color': 'green', 'linestyle': '-.', 'linewidth': 2},
        'Algorithm 3 (Lev Control)': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5}
    }
    
    # Plot each strategy
    for name, returns in cumulative_returns.items():
        style = styles.get(name, {})
        ax.plot(dates, returns, label=name, **style)
    
    # Use log scale for y-axis (better for long-term returns)
    ax.set_yscale('log')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Returns: IVV Volatility Targeting Strategies (2000-2025)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cumulative returns plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_running_volatility(dates: pd.DatetimeIndex,
                            running_vols: Dict[str, np.ndarray],
                            target_vol: float = 15.0,
                            mc_band: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                            save_path: Optional[str] = None) -> None:
    """
    Plot running annualized volatility with Monte Carlo confidence band.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading dates
    running_vols : Dict[str, np.ndarray]
        Dictionary mapping strategy names to running volatility arrays (as %)
    target_vol : float
        Target volatility (as %, e.g., 15.0 for 15%)
    mc_band : Optional[Tuple[np.ndarray, np.ndarray]]
        Monte Carlo (P10, P90) confidence band arrays (as %)
    save_path : Optional[str]
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Monte Carlo confidence band first (in background)
    if mc_band is not None:
        P10, P90 = mc_band
        ax.fill_between(dates, P10, P90, alpha=0.2, color='lightblue', 
                        label='MC 10-90% Confidence Band')
    
    # Plot target volatility line
    ax.axhline(y=target_vol, color='black', linestyle='-', linewidth=2, 
               label=f'Target ({target_vol}%)', alpha=0.8)
    
    # Define colors and line styles
    styles = {
        'IVV Buy-and-Hold': {'color': 'gray', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.7},
        'Algorithm 1 (Open-Loop)': {'color': 'blue', 'linestyle': '--', 'linewidth': 2},
        'Algorithm 2 (Vol Control)': {'color': 'green', 'linestyle': '-.', 'linewidth': 2},
        'Algorithm 3 (Lev Control)': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5}
    }
    
    # Plot running volatility for each strategy
    for name, vols in running_vols.items():
        style = styles.get(name, {})
        ax.plot(dates, vols, label=name, **style)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_title('Running Annualized EWMA Volatility (Halflife=126d) with Monte Carlo Band', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    all_max_values = [np.max(v) for v in running_vols.values()]
    max_vol = max(all_max_values)
    ax.set_ylim(0, max(60, float(max_vol) * 1.1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Running volatility plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_mc_band_standalone(dates: pd.DatetimeIndex,
                            P10: np.ndarray,
                            P50: np.ndarray,
                            P90: np.ndarray,
                            target_vol: float = 15.0,
                            save_path: Optional[str] = None) -> None:
    """
    Plot standalone Monte Carlo confidence band.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading dates
    P10 : np.ndarray
        10th percentile (as %)
    P50 : np.ndarray
        50th percentile (as %)
    P90 : np.ndarray
        90th percentile (as %)
    target_vol : float
        Target volatility (as %)
    save_path : Optional[str]
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Fill between P10 and P90
    ax.fill_between(dates, P10, P90, alpha=0.3, color='lightblue', 
                    label='10-90% Confidence Band')
    
    # Plot median
    ax.plot(dates, P50, color='blue', linestyle='-', linewidth=2, label='Median (P50)')
    
    # Plot target
    ax.axhline(y=target_vol, color='black', linestyle='--', linewidth=2, 
               label=f'True Volatility ({target_vol}%)', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_title('Monte Carlo Confidence Band for EWMA Volatility Estimator\n(10,000 trials, true vol = 15%)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MC band plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_weights_over_time(dates: pd.DatetimeIndex,
                           weights: Dict[str, np.ndarray],
                           save_path: Optional[str] = None) -> None:
    """
    Plot portfolio weights over time for all strategies.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading dates
    weights : Dict[str, np.ndarray]
        Dictionary mapping strategy names to weight arrays
    save_path : Optional[str]
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    styles = {
        'Algorithm 1 (Open-Loop)': {'color': 'blue', 'linestyle': '-', 'linewidth': 1.5},
        'Algorithm 2 (Vol Control)': {'color': 'green', 'linestyle': '-', 'linewidth': 1.5},
        'Algorithm 3 (Lev Control)': {'color': 'red', 'linestyle': '-', 'linewidth': 1.5}
    }
    
    for name, w in weights.items():
        style = styles.get(name, {})
        ax.plot(dates, w, label=name, **style, alpha=0.7)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Fully Invested')
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Leverage Cap')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risky Asset Weight', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Weights Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weights plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_drawdowns(dates: pd.DatetimeIndex,
                   cumulative_returns: Dict[str, np.ndarray],
                   save_path: Optional[str] = None) -> None:
    """
    Plot drawdowns over time for all strategies.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading dates
    cumulative_returns : Dict[str, np.ndarray]
        Dictionary mapping strategy names to cumulative return arrays
    save_path : Optional[str]
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    styles = {
        'IVV Buy-and-Hold': {'color': 'gray', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.7},
        'Algorithm 1 (Open-Loop)': {'color': 'blue', 'linestyle': '--', 'linewidth': 2},
        'Algorithm 2 (Vol Control)': {'color': 'green', 'linestyle': '-.', 'linewidth': 2},
        'Algorithm 3 (Lev Control)': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5}
    }
    
    for name, returns in cumulative_returns.items():
        # Compute running drawdown
        running_max = np.maximum.accumulate(returns)
        drawdowns = (returns / running_max - 1.0) * 100
        
        style = styles.get(name, {})
        ax.plot(dates, drawdowns, label=name, **style)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Drawdowns Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Drawdowns plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
