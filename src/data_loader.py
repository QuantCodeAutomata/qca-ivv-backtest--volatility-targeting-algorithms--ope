"""
Data loading and preprocessing module for IVV backtest.

This module handles:
- Loading IVV adjusted close prices from Yahoo Finance
- Loading Federal Funds Effective Rate from FRED
- Aligning data to common trading calendar
- Computing daily returns
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta


def load_ivv_prices(
    start_date: str = "2000-06-05",  # One day before actual start for r_1 calculation
    end_date: str = "2025-04-09",
    use_adjusted: bool = True
) -> pd.Series:
    """
    Load IVV daily closing prices from Yahoo Finance.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format (should be one day before experiment start)
    end_date : str
        End date in 'YYYY-MM-DD' format
    use_adjusted : bool
        If True, use adjusted close (accounts for splits and dividends)
        If False, use unadjusted close
        
    Returns
    -------
    pd.Series
        Daily closing prices indexed by date
    """
    print(f"Loading IVV prices from {start_date} to {end_date}...")
    
    # Download IVV data
    # Note: yfinance's Close is already adjusted for splits and dividends
    ivv_data = yf.download('IVV', start=start_date, end=end_date, progress=False, auto_adjust=use_adjusted)
    
    # Handle both single and multi-index columns from yfinance
    if isinstance(ivv_data.columns, pd.MultiIndex):
        # MultiIndex format: ('Close', 'IVV')
        prices = ivv_data[('Close', 'IVV')].copy()
    else:
        # Single index format: 'Close'
        prices = ivv_data['Close'].copy()
    
    # Remove any NaN values
    prices = prices.dropna()
    
    print(f"Loaded {len(prices)} trading days of IVV prices")
    
    return prices


def load_fed_funds_rate(
    start_date: str = "2000-06-05",
    end_date: str = "2025-04-09"
) -> pd.Series:
    """
    Load Federal Funds Effective Rate from FRED.
    
    Uses the DFF (daily) series, which is preferred over FEDFUNDS (monthly).
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns
    -------
    pd.Series
        Daily Fed Funds rate (annualized percent) indexed by date
    """
    print(f"Loading Fed Funds rate from {start_date} to {end_date}...")
    
    try:
        # Try DFF first (daily series)
        ff_rate = web.DataReader('DFF', 'fred', start_date, end_date)
        ff_rate = ff_rate['DFF']
    except Exception as e:
        print(f"Failed to load DFF, trying FEDFUNDS: {e}")
        # Fallback to FEDFUNDS (monthly, will need more forward-filling)
        ff_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)
        ff_rate = ff_rate['FEDFUNDS']
    
    print(f"Loaded {len(ff_rate)} observations of Fed Funds rate")
    
    return ff_rate


def align_fed_funds_to_trading_calendar(
    ff_rate: pd.Series,
    trading_dates: pd.DatetimeIndex
) -> pd.Series:
    """
    Align Fed Funds rate to IVV trading calendar using forward-fill.
    
    Parameters
    ----------
    ff_rate : pd.Series
        Fed Funds rate series (may have gaps on weekends/holidays)
    trading_dates : pd.DatetimeIndex
        Trading dates from IVV
        
    Returns
    -------
    pd.Series
        Fed Funds rate aligned to trading calendar, forward-filled
    """
    # Reindex to trading dates and forward-fill
    ff_aligned = ff_rate.reindex(trading_dates, method='ffill')
    
    # Handle any leading NaNs by backward-filling
    ff_aligned = ff_aligned.fillna(method='bfill')
    
    return ff_aligned


def convert_ff_to_daily_return(ff_rate_pct: pd.Series) -> pd.Series:
    """
    Convert annualized Fed Funds rate (in percent) to daily simple return.
    
    Uses ACT/360 convention: r_rf_k = (FFR_k / 100) / 360
    
    Parameters
    ----------
    ff_rate_pct : pd.Series
        Fed Funds rate in annualized percent (e.g., 5.0 for 5%)
        
    Returns
    -------
    pd.Series
        Daily simple return
    """
    return (ff_rate_pct / 100.0) / 360.0


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily simple returns from price series.
    
    r_k = P_k / P_{k-1} - 1
    
    Parameters
    ----------
    prices : pd.Series
        Price series indexed by date
        
    Returns
    -------
    pd.Series
        Simple returns (first observation will be NaN and should be dropped)
    """
    returns = prices.pct_change()
    return returns


def load_and_prepare_data(
    start_date: str = "2000-06-06",
    end_date: str = "2025-04-09",
    use_adjusted: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Load and prepare all data for the backtest.
    
    This is the main entry point for data preparation.
    
    Parameters
    ----------
    start_date : str
        Experiment start date (first date with return data)
    end_date : str
        Experiment end date
    use_adjusted : bool
        Whether to use adjusted close prices for IVV
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        - prices: IVV prices (includes one day before start_date)
        - returns: IVV daily simple returns (starting from start_date)
        - rf_returns: Daily risk-free returns (aligned to returns index)
    """
    # Load IVV prices (one day earlier to compute r_1)
    start_dt = pd.to_datetime(start_date)
    price_start = (start_dt - timedelta(days=5)).strftime('%Y-%m-%d')  # Go back a few days to ensure we get prior close
    
    prices = load_ivv_prices(price_start, end_date, use_adjusted)
    
    # Compute returns
    returns_full = compute_simple_returns(prices)
    returns_full = returns_full.dropna()
    
    # Filter to experiment period
    returns = returns_full[start_date:end_date].copy()
    
    print(f"\nExperiment period: {returns.index[0]} to {returns.index[-1]}")
    print(f"Number of trading days: {len(returns)}")
    
    # Load Fed Funds rate
    ff_rate = load_fed_funds_rate(price_start, end_date)
    
    # Align to trading calendar
    ff_aligned = align_fed_funds_to_trading_calendar(ff_rate, returns.index)
    
    # Convert to daily returns
    rf_returns = convert_ff_to_daily_return(ff_aligned)
    
    # Get corresponding prices for the experiment period
    prices_exp = prices[returns.index]
    
    return prices_exp, returns, rf_returns


if __name__ == "__main__":
    # Test data loading
    prices, returns, rf_returns = load_and_prepare_data()
    
    print("\n=== Data Summary ===")
    print(f"Prices shape: {prices.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"RF returns shape: {rf_returns.shape}")
    print(f"\nFirst date: {returns.index[0]}")
    print(f"Last date: {returns.index[-1]}")
    print(f"\nSample returns:\n{returns.head()}")
    print(f"\nSample RF returns:\n{rf_returns.head()}")
    print(f"\nReturns statistics:")
    print(f"  Mean: {returns.mean():.6f}")
    print(f"  Std: {returns.std():.6f}")
    print(f"  Min: {returns.min():.6f}")
    print(f"  Max: {returns.max():.6f}")
