"""
Trading algorithms implementation.

Implements:
- Algorithm 1: Open-Loop Inverse-Volatility Weighting
- Algorithm 2: Proportional Feedback Volatility Control
- Algorithm 3: Volatility + Leverage Drawdown Control
- Buy-and-Hold IVV Baseline
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass
from src.ewma import compute_ewma_volatility_series, EWMAVolatilityEstimator, EWMAMovingAverage


@dataclass
class StrategyParameters:
    """Parameters for volatility targeting strategies."""
    
    # Target volatility (daily)
    sigma_target: float = 0.15 / np.sqrt(252)
    
    # Leverage cap
    L: float = 1.5
    
    # EWMA halflife (trading days)
    halflife: float = 126.0
    
    # Volatility controller parameters (Algorithm 2 & 3)
    g: float = 50.0  # Controller gain
    kappa_min: float = -1.0
    kappa_max: float = 1.0
    theta: float = 0.5  # Smoothing factor
    
    # Leverage controller parameters (Algorithm 3)
    g_lev: float = 20.0  # Leverage controller gain
    kappa_lev_min: float = -2.0
    h_long: float = 126.0  # Long MA halflife
    h_short: float = 42.0  # Short MA halflife
    
    # Control delay (days before engaging controllers)
    control_delay: int = 10


@dataclass
class StrategyState:
    """State variables for a strategy at time k."""
    
    # Portfolio weights
    w: float = 1.0  # Risky asset weight
    c: float = 0.0  # Cash weight
    
    # Returns
    r_ind: float = 0.0  # Index return
    R_ind: float = 1.0  # Cumulative index level
    
    # Volatility estimates
    sigma_hat_risky: float = 0.0  # EWMA vol of risky asset
    sigma_hat_ind: float = 0.0  # EWMA vol of index
    
    # Controller states (Algorithm 2 & 3)
    kappa: float = 0.0  # Volatility controller output
    e: float = 0.0  # Tracking error
    
    # Leverage controller states (Algorithm 3)
    kappa_lev: float = 0.0  # Leverage controller output
    MA_long: float = 1.0  # Long MA of index level
    MA_short: float = 1.0  # Short MA of index level


class BaseStrategy:
    """Base class for all strategies."""
    
    def __init__(self, params: StrategyParameters):
        """
        Initialize strategy.
        
        Parameters
        ----------
        params : StrategyParameters
            Strategy parameters
        """
        self.params = params
        self.reset()
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.state = StrategyState()
        self.history = []
    
    def update(self, r_risky: float, r_rf: float, k: int) -> StrategyState:
        """
        Update strategy for one time step.
        
        Parameters
        ----------
        r_risky : float
            Risky asset return at time k
        r_rf : float
            Risk-free rate at time k
        k : int
            Time index (1-based)
            
        Returns
        -------
        StrategyState
            Updated state
        """
        raise NotImplementedError("Subclass must implement update()")
    
    def get_history_df(self) -> pd.DataFrame:
        """
        Get history as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            History of all state variables
        """
        return pd.DataFrame([vars(s) for s in self.history])


class BuyAndHoldIVV:
    """
    Buy-and-hold IVV baseline strategy.
    
    Simply holds the risky asset with no rebalancing.
    """
    
    def __init__(self):
        """Initialize buy-and-hold strategy."""
        self.reset()
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.cumulative_returns = []
        self.R = 1.0
    
    def update(self, r_risky: float) -> float:
        """
        Update cumulative return.
        
        Parameters
        ----------
        r_risky : float
            Risky asset return
            
        Returns
        -------
        float
            Cumulative return level
        """
        self.R *= (1.0 + r_risky)
        self.cumulative_returns.append(self.R)
        return self.R
    
    def get_returns(self) -> np.ndarray:
        """Get array of cumulative returns."""
        return np.array(self.cumulative_returns)


class Algorithm1OpenLoop(BaseStrategy):
    """
    Algorithm 1: Open-Loop Inverse-Volatility Weighting.
    
    Weight: w_k = min(σ_target / σ_hat_k, L)
    """
    
    def __init__(self, params: StrategyParameters):
        """Initialize Algorithm 1."""
        self.vol_estimator_risky = EWMAVolatilityEstimator(params.halflife)
        self.vol_estimator_ind = EWMAVolatilityEstimator(params.halflife)
        super().__init__(params)
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        if hasattr(self, 'vol_estimator_risky'):
            self.vol_estimator_risky.reset()
        if hasattr(self, 'vol_estimator_ind'):
            self.vol_estimator_ind.reset()
    
    def update(self, r_risky: float, r_rf: float, k: int) -> StrategyState:
        """Update strategy for one time step."""
        # Step 1: Compute index return using previous weights
        r_ind = r_risky * self.state.w + r_rf * self.state.c
        
        # Step 2: Update cumulative index level
        R_ind = self.state.R_ind * (1.0 + r_ind)
        
        # Step 3: Update EWMA volatility estimates
        sigma_hat_risky = self.vol_estimator_risky.update(r_risky)
        sigma_hat_ind = self.vol_estimator_ind.update(r_ind)
        
        # Step 4: Compute new weight (open-loop)
        if sigma_hat_risky > 0:
            w_new = min(self.params.sigma_target / sigma_hat_risky, self.params.L)
        else:
            # Fallback for edge case
            w_new = self.params.L
        
        # Enforce no shorting
        w_new = max(w_new, 0.0)
        c_new = 1.0 - w_new
        
        # Update state
        self.state = StrategyState(
            w=w_new,
            c=c_new,
            r_ind=r_ind,
            R_ind=R_ind,
            sigma_hat_risky=sigma_hat_risky,
            sigma_hat_ind=sigma_hat_ind,
            kappa=0.0,
            e=0.0,
            kappa_lev=0.0,
            MA_long=R_ind,
            MA_short=R_ind
        )
        
        # Save to history
        self.history.append(self.state)
        
        # Validate
        assert abs(self.state.w + self.state.c - 1.0) < 1e-10, "Weights must sum to 1"
        assert 0.0 <= self.state.w <= self.params.L, f"Weight out of bounds: {self.state.w}"
        
        return self.state


class Algorithm2VolatilityControl(BaseStrategy):
    """
    Algorithm 2: Proportional Feedback Volatility Control.
    
    Adds proportional controller that modulates the open-loop weight:
        e_k = log(σ_hat_k^ind / σ_target)
        κ_k = (1-θ) * clip(-g * e_k, [κ_min, κ_max]) + θ * κ_{k-1}
        w_k = min(exp(κ_k) * σ_target / σ_hat_k, L)
    """
    
    def __init__(self, params: StrategyParameters):
        """Initialize Algorithm 2."""
        self.vol_estimator_risky = EWMAVolatilityEstimator(params.halflife)
        self.vol_estimator_ind = EWMAVolatilityEstimator(params.halflife)
        super().__init__(params)
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        if hasattr(self, 'vol_estimator_risky'):
            self.vol_estimator_risky.reset()
        if hasattr(self, 'vol_estimator_ind'):
            self.vol_estimator_ind.reset()
    
    def update(self, r_risky: float, r_rf: float, k: int) -> StrategyState:
        """Update strategy for one time step."""
        # Step 1: Compute index return using previous weights
        r_ind = r_risky * self.state.w + r_rf * self.state.c
        
        # Step 2: Update cumulative index level
        R_ind = self.state.R_ind * (1.0 + r_ind)
        
        # Step 3: Update EWMA volatility estimates
        sigma_hat_risky = self.vol_estimator_risky.update(r_risky)
        sigma_hat_ind = self.vol_estimator_ind.update(r_ind)
        
        # Step 4: Update volatility controller (if k > control_delay)
        if k > self.params.control_delay and sigma_hat_ind > 0:
            # Tracking error
            e = np.log(sigma_hat_ind / self.params.sigma_target)
            
            # Controller output (before smoothing)
            kappa_raw = -self.params.g * e
            kappa_clipped = np.clip(kappa_raw, self.params.kappa_min, self.params.kappa_max)
            
            # Smooth with previous value
            kappa = (1.0 - self.params.theta) * kappa_clipped + self.params.theta * self.state.kappa
        else:
            e = 0.0
            kappa = 0.0
        
        # Step 5: Compute new weight
        if sigma_hat_risky > 0:
            w_new = min(np.exp(kappa) * self.params.sigma_target / sigma_hat_risky, self.params.L)
        else:
            w_new = self.params.L
        
        # Enforce no shorting
        w_new = max(w_new, 0.0)
        c_new = 1.0 - w_new
        
        # Update state
        self.state = StrategyState(
            w=w_new,
            c=c_new,
            r_ind=r_ind,
            R_ind=R_ind,
            sigma_hat_risky=sigma_hat_risky,
            sigma_hat_ind=sigma_hat_ind,
            kappa=kappa,
            e=e,
            kappa_lev=0.0,
            MA_long=R_ind,
            MA_short=R_ind
        )
        
        # Save to history
        self.history.append(self.state)
        
        # Validate
        assert abs(self.state.w + self.state.c - 1.0) < 1e-10, "Weights must sum to 1"
        assert 0.0 <= self.state.w <= self.params.L, f"Weight out of bounds: {self.state.w}"
        assert self.params.kappa_min <= self.state.kappa <= self.params.kappa_max, f"Kappa out of bounds: {self.state.kappa}"
        
        return self.state


class Algorithm3LeverageControl(BaseStrategy):
    """
    Algorithm 3: Volatility + Leverage Drawdown Control.
    
    Adds second controller that modulates the leverage cap based on MA ratio:
        κ_lev_k = clip(-g_lev * log(MA_long_k / MA_short_k), [κ_lev_min, 0])
        effective_L = exp(κ_lev_k) * L
        w_k = min(exp(κ_k) * σ_target / σ_hat_k, effective_L)
    """
    
    def __init__(self, params: StrategyParameters):
        """Initialize Algorithm 3."""
        self.vol_estimator_risky = EWMAVolatilityEstimator(params.halflife)
        self.vol_estimator_ind = EWMAVolatilityEstimator(params.halflife)
        self.ma_long = EWMAMovingAverage(params.h_long)
        self.ma_short = EWMAMovingAverage(params.h_short)
        super().__init__(params)
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        if hasattr(self, 'vol_estimator_risky'):
            self.vol_estimator_risky.reset()
        if hasattr(self, 'vol_estimator_ind'):
            self.vol_estimator_ind.reset()
        if hasattr(self, 'ma_long'):
            self.ma_long.reset()
        if hasattr(self, 'ma_short'):
            self.ma_short.reset()
    
    def update(self, r_risky: float, r_rf: float, k: int) -> StrategyState:
        """Update strategy for one time step."""
        # Step 1: Compute index return using previous weights
        r_ind = r_risky * self.state.w + r_rf * self.state.c
        
        # Step 2: Update cumulative index level
        R_ind = self.state.R_ind * (1.0 + r_ind)
        
        # Step 3: Update EWMA volatility estimates
        sigma_hat_risky = self.vol_estimator_risky.update(r_risky)
        sigma_hat_ind = self.vol_estimator_ind.update(r_ind)
        
        # Step 4: Update moving averages of index level
        MA_long = self.ma_long.update(R_ind)
        MA_short = self.ma_short.update(R_ind)
        
        # Step 5: Update controllers (if k > control_delay)
        if k > self.params.control_delay:
            # Volatility controller
            if sigma_hat_ind > 0:
                e = np.log(sigma_hat_ind / self.params.sigma_target)
                kappa_raw = -self.params.g * e
                kappa_clipped = np.clip(kappa_raw, self.params.kappa_min, self.params.kappa_max)
                kappa = (1.0 - self.params.theta) * kappa_clipped + self.params.theta * self.state.kappa
            else:
                e = 0.0
                kappa = 0.0
            
            # Leverage controller
            if MA_long > 0 and MA_short > 0:
                kappa_lev = -self.params.g_lev * np.log(MA_long / MA_short)
                kappa_lev = np.clip(kappa_lev, self.params.kappa_lev_min, 0.0)
            else:
                kappa_lev = 0.0
        else:
            e = 0.0
            kappa = 0.0
            kappa_lev = 0.0
        
        # Step 6: Compute new weight with dynamic leverage cap
        effective_L = np.exp(kappa_lev) * self.params.L
        
        if sigma_hat_risky > 0:
            w_new = min(np.exp(kappa) * self.params.sigma_target / sigma_hat_risky, effective_L)
        else:
            w_new = effective_L
        
        # Enforce no shorting
        w_new = max(w_new, 0.0)
        c_new = 1.0 - w_new
        
        # Update state
        self.state = StrategyState(
            w=w_new,
            c=c_new,
            r_ind=r_ind,
            R_ind=R_ind,
            sigma_hat_risky=sigma_hat_risky,
            sigma_hat_ind=sigma_hat_ind,
            kappa=kappa,
            e=e,
            kappa_lev=kappa_lev,
            MA_long=MA_long,
            MA_short=MA_short
        )
        
        # Save to history
        self.history.append(self.state)
        
        # Validate
        assert abs(self.state.w + self.state.c - 1.0) < 1e-10, "Weights must sum to 1"
        assert 0.0 <= self.state.w <= max(self.params.L, effective_L) + 1e-6, f"Weight out of bounds: {self.state.w}"
        assert self.params.kappa_min <= self.state.kappa <= self.params.kappa_max, f"Kappa out of bounds: {self.state.kappa}"
        assert self.params.kappa_lev_min <= self.state.kappa_lev <= 0.0, f"Kappa_lev out of bounds: {self.state.kappa_lev}"
        
        return self.state


def run_backtest(risky_returns: np.ndarray, rf_returns: np.ndarray,
                 strategy_class, params: StrategyParameters) -> pd.DataFrame:
    """
    Run backtest for a given strategy.
    
    Parameters
    ----------
    risky_returns : np.ndarray
        Array of risky asset returns
    rf_returns : np.ndarray
        Array of risk-free returns
    strategy_class : class
        Strategy class to instantiate
    params : StrategyParameters
        Strategy parameters
        
    Returns
    -------
    pd.DataFrame
        DataFrame with full history of strategy states
    """
    strategy = strategy_class(params)
    strategy.reset()
    
    T = len(risky_returns)
    for k in range(T):
        strategy.update(risky_returns[k], rf_returns[k], k + 1)
    
    return strategy.get_history_df()
