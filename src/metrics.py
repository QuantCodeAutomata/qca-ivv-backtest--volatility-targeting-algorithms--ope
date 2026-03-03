"""
Performance metrics computation for backtest results.

Implements:
- Volatility Tracking Error (VTE)
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
"""

import numpy as np
from typing import Dict
import pandas as pd


def compute_volatility_tracking_error(sigma_hat: np.ndarray,
                                      sigma_target_daily: float,
                                      annualization_factor: float = np.sqrt(252)) -> float:
    """
    Compute Volatility Tracking Error (VTE).
    
    VTE = (1/T) * Σ |σ_hat_k - σ_target| * sqrt(252)
    
    Parameters
    ----------
    sigma_hat : np.ndarray
        Running EWMA volatility estimates (daily)
    sigma_target_daily : float
        Target volatility (daily)
    annualization_factor : float
        Annualization factor (default sqrt(252))
        
    Returns
    -------
    float
        Annualized VTE in percent
    """
    T = len(sigma_hat)
    mae = np.mean(np.abs(sigma_hat - sigma_target_daily))
    vte_annualized = mae * annualization_factor
    return vte_annualized * 100  # Convert to percent


def compute_annualized_return(returns: np.ndarray,
                              periods_per_year: int = 252) -> float:
    """
    Compute annualized return (arithmetic mean * periods_per_year).
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    periods_per_year : int
        Number of periods per year (default 252)
        
    Returns
    -------
    float
        Annualized return in percent
    """
    mean_return = np.mean(returns)
    annualized = mean_return * periods_per_year
    return annualized * 100  # Convert to percent


def compute_annualized_volatility(returns: np.ndarray,
                                  periods_per_year: int = 252) -> float:
    """
    Compute annualized volatility (std * sqrt(periods_per_year)).
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    periods_per_year : int
        Number of periods per year (default 252)
        
    Returns
    -------
    float
        Annualized volatility in percent
    """
    std_return = np.std(returns, ddof=1)
    annualized = std_return * np.sqrt(periods_per_year)
    return annualized * 100  # Convert to percent


def compute_sharpe_ratio(annualized_return: float,
                        annualized_volatility: float) -> float:
    """
    Compute Sharpe ratio (no risk-free rate subtraction).
    
    Sharpe = Annualized Return / Annualized Volatility
    
    Note: This matches the paper's Table 1 definition, which does not
    subtract the risk-free rate in the numerator.
    
    Parameters
    ----------
    annualized_return : float
        Annualized return in percent
    annualized_volatility : float
        Annualized volatility in percent
        
    Returns
    -------
    float
        Sharpe ratio
    """
    if annualized_volatility == 0:
        return 0.0
    return annualized_return / annualized_volatility


def compute_maximum_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Compute maximum drawdown.
    
    MaxDD = max_k [1 - R_k / max_{j<=k} R_j]
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Cumulative return levels (R_k, starting from 1.0)
        
    Returns
    -------
    float
        Maximum drawdown in percent
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = 1.0 - cumulative_returns / running_max
    max_dd = np.max(drawdown)
    return max_dd * 100  # Convert to percent


def compute_cagr(cumulative_returns: np.ndarray,
                periods_per_year: int = 252) -> float:
    """
    Compute Compound Annual Growth Rate.
    
    CAGR = (R_T)^(252/T) - 1
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Cumulative return levels
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        CAGR in percent
    """
    T = len(cumulative_returns)
    R_final = cumulative_returns[-1]
    cagr = R_final ** (periods_per_year / T) - 1
    return cagr * 100  # Convert to percent


def compute_all_metrics(results: Dict[str, np.ndarray],
                       sigma_target_daily: float) -> Dict[str, float]:
    """
    Compute all performance metrics for a single strategy.
    
    Parameters
    ----------
    results : dict
        Strategy results containing:
        - 'index_returns': daily returns
        - 'cumulative_returns': cumulative return levels
        - 'sigma_hat': running EWMA volatility
    sigma_target_daily : float
        Target volatility (daily)
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    index_returns = results['index_returns']
    cumulative_returns = results['cumulative_returns']
    sigma_hat = results['sigma_hat']
    
    # Compute metrics
    vte = compute_volatility_tracking_error(sigma_hat, sigma_target_daily)
    ann_return = compute_annualized_return(index_returns)
    ann_vol = compute_annualized_volatility(index_returns)
    sharpe = compute_sharpe_ratio(ann_return, ann_vol)
    max_dd = compute_maximum_drawdown(cumulative_returns)
    cagr = compute_cagr(cumulative_returns)
    
    return {
        'VTE': vte,
        'Annualized_Return': ann_return,
        'Annualized_Volatility': ann_vol,
        'Sharpe_Ratio': sharpe,
        'Maximum_Drawdown': max_dd,
        'CAGR': cagr
    }


def create_performance_table(all_results: Dict[str, Dict[str, np.ndarray]],
                            sigma_target_daily: float) -> pd.DataFrame:
    """
    Create performance metrics table for all strategies.
    
    Parameters
    ----------
    all_results : dict
        Dictionary mapping strategy name to results
    sigma_target_daily : float
        Target volatility (daily)
        
    Returns
    -------
    pd.DataFrame
        Performance metrics table
    """
    metrics_dict = {}
    
    for strategy_name, results in all_results.items():
        metrics = compute_all_metrics(results, sigma_target_daily)
        metrics_dict[strategy_name] = metrics
    
    df = pd.DataFrame(metrics_dict).T
    
    # Reorder columns
    column_order = ['VTE', 'Annualized_Return', 'Annualized_Volatility', 
                   'Sharpe_Ratio', 'Maximum_Drawdown', 'CAGR']
    df = df[column_order]
    
    return df


def format_performance_table(df: pd.DataFrame) -> str:
    """
    Format performance table as markdown for display.
    
    Parameters
    ----------
    df : pd.DataFrame
        Performance metrics table
        
    Returns
    -------
    str
        Formatted markdown table
    """
    # Create a copy for formatting
    df_formatted = df.copy()
    
    # Format each column
    df_formatted['VTE'] = df_formatted['VTE'].apply(lambda x: f"{x:.1f}%")
    df_formatted['Annualized_Return'] = df_formatted['Annualized_Return'].apply(lambda x: f"{x:.1f}%")
    df_formatted['Annualized_Volatility'] = df_formatted['Annualized_Volatility'].apply(lambda x: f"{x:.1f}%")
    df_formatted['Sharpe_Ratio'] = df_formatted['Sharpe_Ratio'].apply(lambda x: f"{x:.2f}")
    df_formatted['Maximum_Drawdown'] = df_formatted['Maximum_Drawdown'].apply(lambda x: f"{x:.1f}%")
    df_formatted['CAGR'] = df_formatted['CAGR'].apply(lambda x: f"{x:.1f}%")
    
    # Rename columns for display
    df_formatted.columns = ['Vol. Tracking Error', 'Ann. Return', 'Ann. Volatility',
                           'Sharpe Ratio', 'Max Drawdown', 'CAGR']
    
    return df_formatted.to_markdown()


def validate_metrics(metrics: Dict[str, float]) -> None:
    """
    Validate computed metrics are in reasonable ranges.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary
        
    Raises
    ------
    ValueError
        If metrics are outside reasonable bounds
    """
    # VTE should be non-negative
    assert metrics['VTE'] >= 0, f"VTE must be non-negative, got {metrics['VTE']}"
    
    # Volatility should be positive
    assert metrics['Annualized_Volatility'] > 0, f"Volatility must be positive, got {metrics['Annualized_Volatility']}"
    
    # Max drawdown should be between 0 and 100%
    assert 0 <= metrics['Maximum_Drawdown'] <= 100, f"Max DD must be in [0, 100], got {metrics['Maximum_Drawdown']}"
    
    # Sharpe ratio should be finite
    assert np.isfinite(metrics['Sharpe_Ratio']), f"Sharpe ratio must be finite, got {metrics['Sharpe_Ratio']}"
