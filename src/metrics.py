"""
Performance metrics calculation.

Implements:
- Volatility Tracking Error (VTE)
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_volatility_tracking_error(sigma_hat_series: np.ndarray, 
                                       sigma_target: float,
                                       annualization_factor: float = np.sqrt(252)) -> float:
    """
    Compute Volatility Tracking Error (VTE).
    
    VTE = (1/T) * Σ |σ_hat_k - σ_target| * annualization_factor
    
    Parameters
    ----------
    sigma_hat_series : np.ndarray
        Running volatility estimates (daily)
    sigma_target : float
        Target volatility (daily)
    annualization_factor : float
        Factor to annualize (default: sqrt(252))
        
    Returns
    -------
    float
        Annualized VTE as percentage
    """
    mae = np.mean(np.abs(sigma_hat_series - sigma_target))
    vte_annualized = mae * annualization_factor
    return vte_annualized * 100  # Convert to percentage


def compute_annualized_return(returns: np.ndarray, 
                               periods_per_year: int = 252) -> float:
    """
    Compute annualized return (arithmetic mean method).
    
    Annual Return = mean(daily returns) * periods_per_year
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    periods_per_year : int
        Number of periods per year (default: 252 trading days)
        
    Returns
    -------
    float
        Annualized return as percentage
    """
    mean_return = np.mean(returns)
    ann_return = mean_return * periods_per_year
    return ann_return * 100  # Convert to percentage


def compute_cagr(cumulative_returns: np.ndarray, 
                 periods_per_year: int = 252) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR).
    
    CAGR = (R_T)^(periods_per_year / T) - 1
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Array of cumulative return levels (starting at 1.0)
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        CAGR as percentage
    """
    T = len(cumulative_returns)
    final_level = cumulative_returns[-1]
    
    cagr = final_level ** (periods_per_year / T) - 1.0
    return cagr * 100  # Convert to percentage


def compute_annualized_volatility(returns: np.ndarray,
                                   periods_per_year: int = 252) -> float:
    """
    Compute annualized volatility.
    
    Annual Vol = std(daily returns) * sqrt(periods_per_year)
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Annualized volatility as percentage
    """
    daily_vol = np.std(returns, ddof=1)
    ann_vol = daily_vol * np.sqrt(periods_per_year)
    return ann_vol * 100  # Convert to percentage


def compute_sharpe_ratio(returns: np.ndarray,
                         periods_per_year: int = 252) -> float:
    """
    Compute Sharpe ratio.
    
    Following the paper's convention (Table 1):
    Sharpe = Annualized Return / Annualized Volatility
    (no risk-free rate subtraction)
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Sharpe ratio
    """
    ann_return = compute_annualized_return(returns, periods_per_year)
    ann_vol = compute_annualized_volatility(returns, periods_per_year)
    
    if ann_vol == 0:
        return 0.0
    
    sharpe = ann_return / ann_vol
    return sharpe


def compute_maximum_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Compute maximum drawdown.
    
    MDD = max_k [1 - R_k / max_{j<=k} R_j]
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Array of cumulative return levels (starting at 1.0)
        
    Returns
    -------
    float
        Maximum drawdown as percentage
    """
    # Compute running maximum
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Compute drawdown at each point
    drawdowns = 1.0 - cumulative_returns / running_max
    
    # Maximum drawdown
    max_dd = np.max(drawdowns)
    return max_dd * 100  # Convert to percentage


def compute_all_metrics(returns: np.ndarray,
                        cumulative_returns: np.ndarray,
                        sigma_hat_series: np.ndarray,
                        sigma_target: float,
                        periods_per_year: int = 252) -> Dict[str, float]:
    """
    Compute all performance metrics for a strategy.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of strategy returns
    cumulative_returns : np.ndarray
        Array of cumulative return levels
    sigma_hat_series : np.ndarray
        Running volatility estimates (daily)
    sigma_target : float
        Target volatility (daily)
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics
    """
    metrics = {
        'VTE': compute_volatility_tracking_error(sigma_hat_series, sigma_target),
        'Ann_Return': compute_annualized_return(returns, periods_per_year),
        'Ann_Volatility': compute_annualized_volatility(returns, periods_per_year),
        'Sharpe': compute_sharpe_ratio(returns, periods_per_year),
        'Max_Drawdown': compute_maximum_drawdown(cumulative_returns),
        'CAGR': compute_cagr(cumulative_returns, periods_per_year),
        'Total_Return': (cumulative_returns[-1] - 1.0) * 100
    }
    
    return metrics


def create_performance_table(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create performance comparison table (Table 1).
    
    Parameters
    ----------
    all_metrics : Dict[str, Dict[str, float]]
        Dictionary mapping strategy names to their metrics
        
    Returns
    -------
    pd.DataFrame
        Performance table
    """
    df = pd.DataFrame(all_metrics).T
    
    # Reorder columns
    column_order = ['VTE', 'Ann_Return', 'Ann_Volatility', 'Sharpe', 'Max_Drawdown', 'CAGR', 'Total_Return']
    df = df[column_order]
    
    # Rename columns for display
    df.columns = ['VTE (%)', 'Ann. Return (%)', 'Ann. Volatility (%)', 
                  'Sharpe Ratio', 'Max Drawdown (%)', 'CAGR (%)', 'Total Return (%)']
    
    return df


def format_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format performance table for display.
    
    Parameters
    ----------
    df : pd.DataFrame
        Performance table
        
    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    # Round to appropriate decimal places
    df_formatted = df.copy()
    
    df_formatted['VTE (%)'] = df_formatted['VTE (%)'].apply(lambda x: f"{float(x):.1f}")
    df_formatted['Ann. Return (%)'] = df_formatted['Ann. Return (%)'].apply(lambda x: f"{float(x):.1f}")
    df_formatted['Ann. Volatility (%)'] = df_formatted['Ann. Volatility (%)'].apply(lambda x: f"{float(x):.1f}")
    df_formatted['Sharpe Ratio'] = df_formatted['Sharpe Ratio'].apply(lambda x: f"{float(x):.2f}")
    df_formatted['Max Drawdown (%)'] = df_formatted['Max Drawdown (%)'].apply(lambda x: f"{float(x):.1f}")
    df_formatted['CAGR (%)'] = df_formatted['CAGR (%)'].apply(lambda x: f"{float(x):.1f}")
    df_formatted['Total Return (%)'] = df_formatted['Total Return (%)'].apply(lambda x: f"{float(x):.1f}")
    
    return df_formatted
