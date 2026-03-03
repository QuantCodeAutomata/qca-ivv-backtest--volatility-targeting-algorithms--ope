"""
Performance metrics calculation for backtest results.

Implements metrics from the paper:
- Volatility Tracking Error (VTE)
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
"""

import numpy as np
from typing import Dict


def volatility_tracking_error(
    sigma_hat_series: np.ndarray,
    sigma_target: float,
    periods_per_year: int = 252
) -> float:
    """
    Compute Volatility Tracking Error (VTE).
    
    VTE = annualized MAE = (1/T) * Σ |σ̂_k - σ_tar| * √252
    
    Parameters
    ----------
    sigma_hat_series : np.ndarray
        Running volatility estimates (daily)
    sigma_target : float
        Target daily volatility
    periods_per_year : int
        Number of trading periods per year
        
    Returns
    -------
    float
        Annualized VTE (as decimal, e.g., 0.052 for 5.2%)
    """
    mae_daily = np.mean(np.abs(sigma_hat_series - sigma_target))
    vte_annualized = mae_daily * np.sqrt(periods_per_year)
    return vte_annualized


def annualized_return(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized return (arithmetic mean method).
    
    Ann. Return = mean(r_k) * 252
    
    Note: This uses arithmetic mean times 252, not CAGR,
    to match the paper's Sharpe ratio calculation.
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    periods_per_year : int
        Number of trading periods per year
        
    Returns
    -------
    float
        Annualized return (as decimal)
    """
    return np.mean(returns) * periods_per_year


def annualized_volatility(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized volatility.
    
    Ann. Vol = std(r_k) * √252
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    periods_per_year : int
        Number of trading periods per year
        
    Returns
    -------
    float
        Annualized volatility (as decimal)
    """
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Compute Sharpe ratio.
    
    Sharpe = Annualized Return / Annualized Volatility
    
    Note: No risk-free rate subtraction, consistent with paper's Table 1.
    This is a return-to-risk ratio rather than excess-return Sharpe.
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    periods_per_year : int
        Number of trading periods per year
        
    Returns
    -------
    float
        Sharpe ratio
    """
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    
    if ann_vol == 0:
        return 0.0
    
    return ann_ret / ann_vol


def maximum_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Compute maximum drawdown.
    
    MaxDD = max_k [1 - R_k / max_{j≤k} R_j]
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Cumulative return series (e.g., starting at 1.0)
        
    Returns
    -------
    float
        Maximum drawdown (as decimal, e.g., 0.553 for 55.3%)
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = 1.0 - cumulative_returns / running_max
    max_dd = np.max(drawdowns)
    return max_dd


def cagr(
    cumulative_returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR).
    
    CAGR = (R_T)^(252/T) - 1
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Cumulative return series
    periods_per_year : int
        Number of trading periods per year
        
    Returns
    -------
    float
        CAGR (as decimal)
    """
    T = len(cumulative_returns)
    final_value = cumulative_returns[-1]
    
    if final_value <= 0 or T == 0:
        return 0.0
    
    cagr_value = final_value ** (periods_per_year / T) - 1.0
    return cagr_value


def compute_all_metrics(
    returns_index: np.ndarray,
    cumulative_index: np.ndarray,
    sigma_hat_series: np.ndarray,
    sigma_target: float,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Compute all performance metrics for a strategy.
    
    Parameters
    ----------
    returns_index : np.ndarray
        Daily index returns
    cumulative_index : np.ndarray
        Cumulative index levels
    sigma_hat_series : np.ndarray
        Running volatility estimates (daily)
    sigma_target : float
        Target daily volatility
    periods_per_year : int
        Number of trading periods per year
        
    Returns
    -------
    Dict[str, float]
        Dictionary of all metrics
    """
    metrics = {
        'VTE': volatility_tracking_error(sigma_hat_series, sigma_target, periods_per_year),
        'Annualized Return': annualized_return(returns_index, periods_per_year),
        'Annualized Volatility': annualized_volatility(returns_index, periods_per_year),
        'Sharpe Ratio': sharpe_ratio(returns_index, periods_per_year),
        'Maximum Drawdown': maximum_drawdown(cumulative_index),
        'CAGR': cagr(cumulative_index, periods_per_year),
    }
    
    return metrics


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """
    Format metrics as a markdown table.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, float]]
        Dictionary mapping strategy name to metrics dictionary
        
    Returns
    -------
    str
        Markdown formatted table
    """
    # Get strategy names
    strategies = list(metrics_dict.keys())
    
    # Get metric names (assume all strategies have same metrics)
    metric_names = list(metrics_dict[strategies[0]].keys())
    
    # Build table header
    header = "| Metric | " + " | ".join(strategies) + " |"
    separator = "|--------|" + "|".join(["--------"] * len(strategies)) + "|"
    
    lines = [header, separator]
    
    # Build table rows
    for metric_name in metric_names:
        row_values = [metrics_dict[strategy][metric_name] for strategy in strategies]
        
        # Format values based on metric type
        if metric_name in ['VTE', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown', 'CAGR']:
            formatted_values = [f"{val*100:.2f}%" for val in row_values]
        else:  # Sharpe Ratio
            formatted_values = [f"{val:.3f}" for val in row_values]
        
        row = f"| {metric_name} | " + " | ".join(formatted_values) + " |"
        lines.append(row)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    print("=== Testing Metrics ===\n")
    
    np.random.seed(42)
    T = 252 * 10  # 10 years
    
    # Synthetic strategy 1: target vol
    sigma_target = 0.15 / np.sqrt(252)
    returns1 = np.random.normal(0.0004, sigma_target, T)
    cumulative1 = np.cumprod(1 + returns1)
    sigma_hat1 = np.ones(T) * sigma_target  # Perfect tracking
    
    # Synthetic strategy 2: higher vol
    returns2 = np.random.normal(0.0005, 0.20 / np.sqrt(252), T)
    cumulative2 = np.cumprod(1 + returns2)
    sigma_hat2 = np.ones(T) * 0.20 / np.sqrt(252)
    
    # Compute metrics
    metrics1 = compute_all_metrics(returns1, cumulative1, sigma_hat1, sigma_target)
    metrics2 = compute_all_metrics(returns2, cumulative2, sigma_hat2, sigma_target)
    
    print("Strategy 1 (15% vol target):")
    for metric, value in metrics1.items():
        if metric == 'Sharpe Ratio':
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value*100:.2f}%")
    
    print("\nStrategy 2 (20% vol):")
    for metric, value in metrics2.items():
        if metric == 'Sharpe Ratio':
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value*100:.2f}%")
    
    # Test table formatting
    print("\n=== Formatted Table ===\n")
    metrics_dict = {
        'Strategy 1': metrics1,
        'Strategy 2': metrics2,
    }
    table = format_metrics_table(metrics_dict)
    print(table)
