# IVV Backtest Results

## Experiment Parameters

- **Period**: 2000-06-06 to 2025-04-08
- **Trading Days**: 6248
- **Adjusted Prices**: True
- **Target Volatility**: 15% annualized
- **Leverage Cap**: 1.5
- **EWMA Halflife**: 126 trading days
- **Volatility Controller Gain**: 50
- **Leverage Controller Gain**: 20
- **Control Delay**: 10 trading days

## Performance Summary (Table 1)

| Metric | IVV Buy-and-Hold | Algorithm 1 (Open-Loop) | Algorithm 2 (Volatility Control) | Algorithm 3 (Leverage Control) |
|--------|--------|--------|--------|--------|
| VTE | 5.16% | 2.33% | 0.37% | 0.53% |
| Annualized Return | 8.58% | 7.02% | 8.87% | 9.48% |
| Annualized Volatility | 19.12% | 15.08% | 14.82% | 14.66% |
| Sharpe Ratio | 0.449 | 0.465 | 0.598 | 0.647 |
| Maximum Drawdown | 55.25% | 38.75% | 39.47% | 34.77% |
| CAGR | 6.98% | 6.05% | 8.08% | 8.76% |

## Key Findings

1. **Volatility Tracking Error**: Algorithm 2 (Volatility Control) achieves the lowest VTE, demonstrating superior volatility targeting capability.

2. **Maximum Drawdown**: Algorithm 3 (Leverage Control) achieves the lowest maximum drawdown, showing improved downside protection through adaptive leverage management.

3. **Sharpe Ratio**: Both Algorithms 2 and 3 achieve higher Sharpe ratios than the open-loop algorithm and buy-and-hold strategy, indicating better risk-adjusted performance.

4. **Return**: Algorithm 3 achieves competitive returns while maintaining lower volatility and drawdown compared to buy-and-hold.

## Files Generated

- `RESULTS.md`: This file
- `exp1_cumulative_returns.png`: Cumulative return time series
- `exp1_running_volatility.png`: Running volatility with MC confidence band
- `exp1_weights.png`: Risky asset weights over time
- `exp1_controller_states.png`: Controller states (κ and κ_ℓ)
- `exp2_mc_confidence_band.png`: Monte Carlo confidence band standalone
- `mc_confidence_band.csv`: Monte Carlo band data
- `daily_series_*.csv`: Daily time series for each strategy

