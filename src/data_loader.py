"""
Data loading and preprocessing for IVV backtest.

This module handles:
- Downloading IVV daily adjusted close prices
- Downloading Federal Funds Effective Rate from FRED
- Aligning data to trading calendar
- Computing daily returns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from typing import Tuple
import os


def load_ivv_data(start_date: str = "2000-06-05", 
                  end_date: str = "2025-04-09",
                  use_adjusted: bool = True,
                  cache_dir: str = "data") -> pd.DataFrame:
    """
    Load IVV daily price data from Yahoo Finance.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format (one day before actual start for return calculation)
    end_date : str
        End date in YYYY-MM-DD format
    use_adjusted : bool
        If True, use adjusted close (split and dividend adjusted)
        If False, use unadjusted close
    cache_dir : str
        Directory to cache downloaded data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date (index), Close, Returns
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"ivv_{'adj' if use_adjusted else 'raw'}_{start_date}_{end_date}.csv")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"Loading IVV data from cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df
    
    print(f"Downloading IVV data from Yahoo Finance ({start_date} to {end_date})...")
    
    # Download data
    ivv = yf.download("IVV", start=start_date, end=end_date, progress=False)
    
    if ivv.empty:
        raise ValueError("No IVV data downloaded. Check date range and internet connection.")
    
    # Handle MultiIndex columns from yfinance
    if isinstance(ivv.columns, pd.MultiIndex):
        ivv.columns = ivv.columns.droplevel(1)
    
    # Select close column
    if use_adjusted:
        close_col = 'Adj Close' if 'Adj Close' in ivv.columns else 'Close'
    else:
        close_col = 'Close'
    
    df = pd.DataFrame({
        'Close': ivv[close_col].values
    }, index=ivv.index)
    
    # Compute daily simple returns: r_k = P_k / P_{k-1} - 1
    df['Returns'] = df['Close'].pct_change()
    
    # Remove the first row (used only for computing first return)
    df = df.iloc[1:].copy()
    
    print(f"Loaded {len(df)} trading days of IVV data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Cache the data
    df.to_csv(cache_file)
    
    return df


def load_fed_funds_rate(start_date: str = "2000-06-05",
                        end_date: str = "2025-04-09",
                        cache_dir: str = "data") -> pd.Series:
    """
    Load Federal Funds Effective Rate from FRED.
    
    Uses DFF (daily) series. Forward-fills missing values to align with trading calendar.
    Converts annualized percent rate to daily return using ACT/360 convention.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    cache_dir : str
        Directory to cache downloaded data
        
    Returns
    -------
    pd.Series
        Daily risk-free rate returns (not annualized percent)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"fedfunds_{start_date}_{end_date}.csv")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"Loading Fed Funds data from cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True, squeeze=True)
        return df['DailyReturn'] if isinstance(df, pd.DataFrame) else df
    
    print(f"Downloading Federal Funds Rate from FRED ({start_date} to {end_date})...")
    
    # Try DFF (daily) first
    try:
        ff_rate = pdr.DataReader('DFF', 'fred', start_date, end_date)
        series_name = 'DFF'
    except Exception as e:
        print(f"DFF not available, trying FEDFUNDS: {e}")
        try:
            ff_rate = pdr.DataReader('FEDFUNDS', 'fred', start_date, end_date)
            series_name = 'FEDFUNDS'
        except Exception as e:
            raise ValueError(f"Could not download Fed Funds data: {e}")
    
    # Forward fill missing values
    ff_rate = ff_rate[series_name].ffill()
    
    # Convert annualized percent to daily return using ACT/360
    # r_rf_k = (FFR_k / 100) / 360
    daily_return = (ff_rate / 100.0) / 360.0
    
    print(f"Loaded {len(daily_return)} days of Fed Funds data")
    
    # Cache the data
    df_cache = pd.DataFrame({'DailyReturn': daily_return})
    df_cache.to_csv(cache_file)
    
    return daily_return


def align_data(ivv_df: pd.DataFrame, 
               rf_series: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Align IVV returns and risk-free rate to common trading calendar.
    
    Parameters
    ----------
    ivv_df : pd.DataFrame
        IVV data with Returns column
    rf_series : pd.Series
        Daily risk-free rate returns
        
    Returns
    -------
    tuple
        (aligned_df, risky_returns, rf_returns)
        - aligned_df: DataFrame with Date, Close, Returns, RF_Rate
        - risky_returns: numpy array of IVV returns
        - rf_returns: numpy array of risk-free returns
    """
    # Reindex risk-free rate to IVV trading calendar and forward-fill
    rf_aligned = rf_series.reindex(ivv_df.index, method='ffill')
    
    # If still NaN at beginning, backfill
    rf_aligned = rf_aligned.fillna(method='bfill')
    
    # Create aligned DataFrame
    aligned_df = ivv_df.copy()
    aligned_df['RF_Rate'] = rf_aligned
    
    # Extract numpy arrays
    risky_returns = aligned_df['Returns'].values
    rf_returns = aligned_df['RF_Rate'].values
    
    # Validate no NaN values
    if np.any(np.isnan(risky_returns)):
        raise ValueError("NaN values found in risky returns")
    if np.any(np.isnan(rf_returns)):
        raise ValueError("NaN values found in risk-free returns")
    
    print(f"Aligned data: {len(aligned_df)} trading days")
    print(f"First date: {aligned_df.index[0]}")
    print(f"Last date: {aligned_df.index[-1]}")
    print(f"Mean risky return: {np.mean(risky_returns):.6f}")
    print(f"Mean RF rate: {np.mean(rf_returns):.6f}")
    
    return aligned_df, risky_returns, rf_returns


def prepare_backtest_data(start_date: str = "2000-06-06",
                          end_date: str = "2025-04-09",
                          use_adjusted: bool = True,
                          cache_dir: str = "data") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare all data needed for backtest.
    
    Parameters
    ----------
    start_date : str
        First trading date for backtest
    end_date : str
        Last trading date for backtest
    use_adjusted : bool
        Use adjusted close prices
    cache_dir : str
        Cache directory
        
    Returns
    -------
    tuple
        (aligned_df, risky_returns, rf_returns)
    """
    # Load IVV (start one day earlier for return calculation)
    start_download = pd.to_datetime(start_date) - pd.Timedelta(days=10)  # Buffer for weekends
    start_download_str = start_download.strftime("%Y-%m-%d")
    
    ivv_df = load_ivv_data(start_download_str, end_date, use_adjusted, cache_dir)
    
    # Filter to actual start date
    ivv_df = ivv_df[ivv_df.index >= start_date]
    
    # Load Fed Funds
    rf_series = load_fed_funds_rate(start_download_str, end_date, cache_dir)
    
    # Align data
    aligned_df, risky_returns, rf_returns = align_data(ivv_df, rf_series)
    
    return aligned_df, risky_returns, rf_returns
