# IVV Backtest: Volatility Targeting Algorithms

This repository implements a comprehensive backtesting framework for volatility targeting strategies on the iShares Core S&P 500 ETF (IVV) from June 6, 2000 to April 9, 2025.

## Overview

The project reproduces two key experiments:

1. **Experiment 1**: Full IVV backtest comparing four strategies:
   - IVV Buy-and-Hold (baseline)
   - Algorithm 1: Open-Loop Inverse-Volatility Weighting
   - Algorithm 2: Proportional Feedback Volatility Control
   - Algorithm 3: Volatility + Leverage Drawdown Control

2. **Experiment 2**: Monte Carlo confidence band generation for EWMA volatility estimator

## Strategies

### Algorithm 1: Open-Loop Inverse-Volatility Weighting
- Weight: `w_k = min(σ_target / σ_hat_k, L)`
- Target volatility: 15% annualized
- Leverage cap: 1.5x

### Algorithm 2: Proportional Feedback Volatility Control
- Adds proportional controller: `κ_k` to modulate open-loop weight
- Controller gain: g = 50
- Smoothing factor: θ = 0.5

### Algorithm 3: Volatility + Leverage Drawdown Control
- Adds second controller for dynamic leverage cap
- Based on long/short EWMA ratio of cumulative index level
- Leverage controller gain: g_ℓ = 20

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data retrieval and preprocessing
│   ├── ewma.py                  # EWMA volatility estimation
│   ├── algorithms.py            # Trading algorithms implementation
│   ├── metrics.py               # Performance metrics
│   ├── monte_carlo.py           # Monte Carlo simulation
│   └── visualization.py         # Plotting functions
├── tests/
│   ├── __init__.py
│   ├── test_ewma.py
│   ├── test_algorithms.py
│   ├── test_metrics.py
│   └── test_monte_carlo.py
├── results/
│   ├── RESULTS.md               # Summary of findings
│   ├── table1_performance.csv   # Performance metrics table
│   ├── cumulative_returns.png   # Cumulative returns plot
│   └── running_volatility.png   # Running volatility plot
├── run_experiments.py           # Main experiment runner
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Set your Massive API key:
```bash
export MASSIVE_TOKEN="your_api_key_here"
```

Run the full experiment:
```bash
python run_experiments.py
```

Run tests:
```bash
pytest tests/ -v
```

## Results

All results are saved in the `results/` directory:
- `RESULTS.md`: Detailed summary of findings and metrics
- `table1_performance.csv`: Performance comparison table
- `cumulative_returns.png`: Cumulative return time series
- `running_volatility.png`: Running annualized volatility with Monte Carlo bands

## Dependencies

- numpy, pandas, scipy
- matplotlib, seaborn
- massive (for financial data)
- pandas-datareader (for Fed Funds rate)
- pytest (for testing)

## Methodology

### Data
- **IVV Daily Prices**: Adjusted closing prices from June 6, 2000 to April 9, 2025
- **Fed Funds Rate**: Daily effective rate (FRED DFF series), ACT/360 convention

### Parameters
- Target daily volatility: σ_tar = 0.15 / √252
- Leverage cap: L = 1.5
- EWMA halflife: h = 126 trading days
- Volatility controller gain: g = 50
- Leverage controller gain: g_ℓ = 20

### Performance Metrics
- Volatility Tracking Error (VTE)
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown

## License

MIT License
