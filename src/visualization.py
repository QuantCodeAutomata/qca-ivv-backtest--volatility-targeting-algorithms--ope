"""
Visualization functions for backtest results.

Creates:
1. Cumulative returns plot (four strategies)
2. Running volatility plot with confidence bands
3. Monte Carlo band standalone plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import matplotlib.dates as mdates


def plot_cumulative_returns(all_results: Dict[str, Dict[str, np.ndarray]],
                           dates: pd.DatetimeIndex,
                           filename: str = "cumulative_returns.png",
                           figsize: tuple = (12, 6)) -> None:
    """
    Plot cumulative returns for all four strategies.
    
    Parameters
    ----------
    all_results : dict
        Dictionary mapping strategy name to results
    dates : pd.DatetimeIndex
        Trading dates
    filename : str
        Output filename
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors and labels
    strategy_config = {
        'IVV': {'color': 'black', 'label': 'IVV Buy-and-Hold', 'linestyle': '-', 'linewidth': 2},
        'Algorithm_1': {'color': 'blue', 'label': 'Algorithm 1 (Open-Loop)', 'linestyle': '--', 'linewidth': 1.5},
        'Algorithm_2': {'color': 'green', 'label': 'Algorithm 2 (Vol Control)', 'linestyle': '-.', 'linewidth': 1.5},
        'Algorithm_3': {'color': 'red', 'label': 'Algorithm 3 (Vol + Lev Control)', 'linestyle': '-', 'linewidth': 2}
    }
    
    # Plot each strategy
    for strategy_name in ['IVV', 'Algorithm_1', 'Algorithm_2', 'Algorithm_3']:
        cumulative_returns = all_results[strategy_name]['cumulative_returns']
        config = strategy_config[strategy_name]
        
        ax.plot(dates, cumulative_returns,
               color=config['color'],
               label=config['label'],
               linestyle=config['linestyle'],
               linewidth=config['linewidth'])
    
    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (log scale)', fontsize=12)
    ax.set_title('Cumulative Returns: IVV vs Volatility Targeting Algorithms', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved cumulative returns plot to {filename}")
    plt.close()


def plot_running_volatility(all_results: Dict[str, Dict[str, np.ndarray]],
                           dates: pd.DatetimeIndex,
                           monte_carlo_band: Optional[Dict[str, np.ndarray]] = None,
                           sigma_target_annualized: float = 15.0,
                           filename: str = "running_volatility.png",
                           figsize: tuple = (12, 6)) -> None:
    """
    Plot running annualized EWMA volatility for all strategies.
    
    Includes:
    - Four volatility lines
    - 15% target horizontal line
    - Optional Monte Carlo confidence band overlay
    
    Parameters
    ----------
    all_results : dict
        Dictionary mapping strategy name to results
    dates : pd.DatetimeIndex
        Trading dates
    monte_carlo_band : dict, optional
        Monte Carlo band with P10, P50, P90
    sigma_target_annualized : float
        Target volatility (annualized, in percent)
    filename : str
        Output filename
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot Monte Carlo confidence band first (if provided)
    if monte_carlo_band is not None:
        ax.fill_between(dates,
                        monte_carlo_band['P10'],
                        monte_carlo_band['P90'],
                        color='lightgray',
                        alpha=0.4,
                        label='Monte Carlo 10-90% Band')
    
    # Plot target volatility line
    ax.axhline(y=sigma_target_annualized, color='darkgray', linestyle='--',
              linewidth=2, label='15% Target', zorder=1)
    
    # Define colors and labels
    strategy_config = {
        'IVV': {'color': 'black', 'label': 'IVV Buy-and-Hold', 'linestyle': '-', 'linewidth': 2},
        'Algorithm_1': {'color': 'blue', 'label': 'Algorithm 1', 'linestyle': '--', 'linewidth': 1.5},
        'Algorithm_2': {'color': 'green', 'label': 'Algorithm 2', 'linestyle': '-.', 'linewidth': 1.5},
        'Algorithm_3': {'color': 'red', 'label': 'Algorithm 3', 'linestyle': '-', 'linewidth': 2}
    }
    
    # Plot each strategy's running volatility
    for strategy_name in ['IVV', 'Algorithm_1', 'Algorithm_2', 'Algorithm_3']:
        sigma_hat = all_results[strategy_name]['sigma_hat']
        sigma_hat_annualized_pct = sigma_hat * np.sqrt(252) * 100
        config = strategy_config[strategy_name]
        
        ax.plot(dates, sigma_hat_annualized_pct,
               color=config['color'],
               label=config['label'],
               linestyle=config['linestyle'],
               linewidth=config['linewidth'],
               alpha=0.8,
               zorder=2)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax.set_title('Running Annualized EWMA Volatility (Halflife = 126 days)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis limits
    ax.set_ylim(0, max(50, sigma_target_annualized * 2))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved running volatility plot to {filename}")
    plt.close()


def plot_monte_carlo_band(band: Dict[str, np.ndarray],
                         sigma_target_annualized: float = 15.0,
                         filename: str = "monte_carlo_band.png",
                         figsize: tuple = (12, 6)) -> None:
    """
    Plot Monte Carlo confidence band standalone.
    
    Parameters
    ----------
    band : dict
        Band dictionary with P10, P50, P90
    sigma_target_annualized : float
        Target volatility (annualized, in percent)
    filename : str
        Output filename
    figsize : tuple
        Figure size
    """
    T = len(band['P50'])
    time_index = np.arange(1, T + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confidence band
    ax.fill_between(time_index, band['P10'], band['P90'],
                   color='lightblue', alpha=0.5, label='10-90% Confidence Band')
    
    # Plot median
    ax.plot(time_index, band['P50'], color='blue', linewidth=2,
           label='Median (50th percentile)')
    
    # Plot target
    ax.axhline(y=sigma_target_annualized, color='red', linestyle='--',
              linewidth=2, label='True Volatility (15%)')
    
    # Formatting
    ax.set_xlabel('Trading Day', fontsize=12)
    ax.set_ylabel('Annualized Volatility Estimate (%)', fontsize=12)
    ax.set_title('Monte Carlo Confidence Band for EWMA Volatility Estimator\n(True Vol = 15%, Halflife = 126 days, 10,000 trials)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, T)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved Monte Carlo band plot to {filename}")
    plt.close()


def plot_weights_evolution(results: Dict[str, np.ndarray],
                          dates: pd.DatetimeIndex,
                          strategy_name: str,
                          filename: str = "weights.png",
                          figsize: tuple = (12, 6)) -> None:
    """
    Plot evolution of portfolio weights over time.
    
    Parameters
    ----------
    results : dict
        Strategy results
    dates : pd.DatetimeIndex
        Trading dates
    strategy_name : str
        Strategy name for title
    filename : str
        Output filename
    figsize : tuple
        Figure size
    """
    weights = results['weights']
    cash = results['cash']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot weights
    ax.plot(dates, weights, label='Risky Asset Weight', color='blue', linewidth=1.5)
    ax.plot(dates, cash, label='Cash Weight', color='green', linewidth=1.5)
    
    # Add leverage cap line
    ax.axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='Leverage Cap')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='100% Invested')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title(f'Portfolio Weights Evolution: {strategy_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved weights plot to {filename}")
    plt.close()


def create_all_plots(all_results: Dict[str, Dict[str, np.ndarray]],
                    dates: pd.DatetimeIndex,
                    monte_carlo_band: Optional[Dict[str, np.ndarray]] = None,
                    output_dir: str = "results") -> None:
    """
    Create all plots for the backtest.
    
    Parameters
    ----------
    all_results : dict
        Dictionary mapping strategy name to results
    dates : pd.DatetimeIndex
        Trading dates
    monte_carlo_band : dict, optional
        Monte Carlo confidence band
    output_dir : str
        Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Cumulative returns
    plot_cumulative_returns(all_results, dates,
                           filename=os.path.join(output_dir, "cumulative_returns.png"))
    
    # Plot 2: Running volatility with Monte Carlo band
    plot_running_volatility(all_results, dates, monte_carlo_band,
                           filename=os.path.join(output_dir, "running_volatility.png"))
    
    # Plot 3: Monte Carlo band standalone (if available)
    if monte_carlo_band is not None:
        plot_monte_carlo_band(monte_carlo_band,
                            filename=os.path.join(output_dir, "monte_carlo_band.png"))
    
    # Plot 4: Weights evolution for each algorithm
    for strategy_name in ['Algorithm_1', 'Algorithm_2', 'Algorithm_3']:
        plot_weights_evolution(all_results[strategy_name], dates, strategy_name,
                             filename=os.path.join(output_dir, f"weights_{strategy_name}.png"))
    
    print(f"All plots saved to {output_dir}/")
