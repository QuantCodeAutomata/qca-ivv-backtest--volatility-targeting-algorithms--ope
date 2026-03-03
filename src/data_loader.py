"""
Data loading and preprocessing module for IVV backtest.

This module handles:
- Loading IVV adjusted closing prices
- Loading Federal Funds Effective Rate
- Aligning data to trading calendar
- Computing daily returns
"""

import os
from typing import Tuple
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf


def load_ivv_data(start_date: str = "2000-06-06", end_date: str = "2025-04-09", 
                  adjusted: bool = True) -> pd.DataFrame:
    """
    Load IVV daily closing prices.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    adjusted : bool
        If True, use adjusted close (accounts for splits and dividends)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Date index and 'Close' column
    """
    print(f"Loading IVV data from {start_date} to {end_date} (adjusted={adjusted})...")
    
    # We need one extra day before start_date to compute first return
    start_extended = pd.to_datetime(start_date) - pd.Timedelta(days=10)
    
    # Download data using yfinance
    data = yf.download('IVV', start=start_extended.strftime('%Y-%m-%d'), 
                       end=end_date, progress=False)
    
    if data.empty:
        raise ValueError("No data retrieved for IVV")
    
    # Select appropriate price column
    if adjusted:
        if 'Adj Close' in data.columns:
            prices = data[['Adj Close']].copy()
            prices.columns = ['Close']
        else:
            prices = data[['Close']].copy()
    else:
        prices = data[['Close']].copy()
    
    # Remove any NaN values
    prices = prices.dropna()
    
    print(f"Loaded {len(prices)} days of IVV data")
    return prices


def load_fed_funds_rate(start_date: str = "2000-06-06", 
                        end_date: str = "2025-04-09") -> pd.Series:
    """
    Load Federal Funds Effective Rate from FRED.
    
    Uses DFF (daily) series. Forward-fills missing values to align with trading days.
    Converts annualized percent rate to daily simple return using ACT/360 convention.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
        
    Returns
    -------
    pd.Series
        Daily risk-free rate (as decimal daily return)
    """
    print(f"Loading Fed Funds rate from {start_date} to {end_date}...")
    
    try:
        # Try DFF (daily) first
        ff_rate = pdr.get_data_fred('DFF', start=start_date, end=end_date)
    except Exception as e:
        print(f"Failed to load DFF: {e}")
        try:
            # Fallback to FEDFUNDS (monthly)
            ff_rate = pdr.get_data_fred('FEDFUNDS', start=start_date, end=end_date)
        except Exception as e2:
            print(f"Failed to load FEDFUNDS: {e2}")
            # If both fail, return zeros
            print("Warning: Using zero risk-free rate")
            return pd.Series(0.0, index=pd.date_range(start_date, end_date))
    
    # Forward fill to handle weekends/holidays
    ff_rate = ff_rate.ffill()
    
    # Convert from annualized percent to daily decimal return using ACT/360
    # r_rf_k = (FFR_k / 100) / 360
    ff_daily = (ff_rate.iloc[:, 0] / 100.0) / 360.0
    
    print(f"Loaded {len(ff_daily)} days of Fed Funds data")
    return ff_daily


def align_data(ivv_prices: pd.DataFrame, fed_funds: pd.Series, 
               start_date: str = "2000-06-06") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align IVV prices and Fed Funds rate to common trading calendar.
    
    The trading calendar is driven by IVV trading dates.
    Fed Funds rate is forward-filled to match IVV dates.
    
    Parameters
    ----------
    ivv_prices : pd.DataFrame
        IVV prices with Date index
    fed_funds : pd.Series
        Fed Funds daily rate
    start_date : str
        First date to include in aligned data
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Aligned (ivv_prices, fed_funds_rate)
    """
    # Filter to start date and after
    start_dt = pd.to_datetime(start_date)
    ivv_prices = ivv_prices[ivv_prices.index >= start_dt].copy()
    
    # Reindex fed funds to match IVV trading dates, forward-fill
    fed_funds_aligned = fed_funds.reindex(ivv_prices.index, method='ffill')
    
    # Fill any remaining NaNs with 0
    fed_funds_aligned = fed_funds_aligned.fillna(0.0)
    
    print(f"Aligned data: {len(ivv_prices)} trading days from {ivv_prices.index[0].date()} to {ivv_prices.index[-1].date()}")
    
    return ivv_prices, fed_funds_aligned


def compute_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Compute daily simple returns from price series.
    
    r_k = P_k / P_{k-1} - 1
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with 'Close' column
        
    Returns
    -------
    pd.Series
        Daily simple returns (first value is NaN)
    """
    returns = prices['Close'].pct_change()
    return returns


def load_and_prepare_data(start_date: str = "2000-06-06", 
                          end_date: str = "2025-04-09",
                          adjusted: bool = True) -> Tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """
    Load and prepare all data for backtesting.
    
    This is the main entry point for data loading. It:
    1. Loads IVV prices
    2. Loads Fed Funds rate
    3. Aligns to common calendar
    4. Computes returns
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    adjusted : bool
        Whether to use adjusted prices
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.DatetimeIndex]
        (ivv_returns, fed_funds_rate, dates)
        Note: ivv_returns[0] is the return FROM the day before start_date TO start_date
    """
    # Load data
    ivv_prices = load_ivv_data(start_date, end_date, adjusted=adjusted)
    fed_funds = load_fed_funds_rate(start_date, end_date)
    
    # Align to common calendar
    ivv_prices_aligned, fed_funds_aligned = align_data(ivv_prices, fed_funds, start_date)
    
    # Compute returns
    ivv_returns = compute_returns(ivv_prices_aligned)
    
    # Drop the first NaN return
    ivv_returns = ivv_returns.dropna()
    fed_funds_aligned = fed_funds_aligned[ivv_returns.index]
    
    dates = ivv_returns.index
    
    print(f"\nData preparation complete:")
    print(f"  Trading days: {len(dates)}")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  Mean daily return: {float(ivv_returns.mean()):.6f}")
    print(f"  Daily volatility: {float(ivv_returns.std()):.6f}")
    print(f"  Mean Fed Funds rate: {float(fed_funds_aligned.mean()):.6f}")
    
    return ivv_returns, fed_funds_aligned, dates
