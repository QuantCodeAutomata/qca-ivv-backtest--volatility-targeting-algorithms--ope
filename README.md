# IVV Volatility Targeting Algorithms Backtest

This repository implements a comprehensive backtest comparing volatility targeting algorithms for the IVV ETF (iShares Core S&P 500 ETF) from June 6, 2000 to April 9, 2025.

## Overview

This project reproduces two key experiments:

1. **Experiment 1**: Full IVV backtest comparing four strategies:
   - IVV Buy-and-Hold (baseline benchmark)
   - Algorithm 1: Open-loop inverse-volatility weighting
   - Algorithm 2: Proportional feedback volatility control
   - Algorithm 3: Volatility + leverage drawdown control

2. **Experiment 2**: Monte Carlo confidence band analysis for EWMA volatility estimator

## Algorithms

### Algorithm 1: Open-Loop Inverse-Volatility Weighting
- Weight: `w_k = min(σ_target / σ_hat_k, L)`
- Target volatility: 15% annualized
- Leverage cap: 1.5
- EWMA halflife: 126 trading days

### Algorithm 2: Proportional Feedback Volatility Control
- Adds proportional feedback controller to modulate open-loop weight
- Controller gain: g = 50
- Updates kappa_k based on tracking error
- Smoothing parameter: θ = 0.5

### Algorithm 3: Volatility + Leverage Drawdown Control
- Extends Algorithm 2 with dynamic leverage cap
- Uses long/short EWMA ratio of cumulative index level
- Leverage controller gain: g_ell = 20
- MA halflives: h_long = 126, h_short = 42

## Data Requirements

- **IVV Daily Prices**: Total-return adjusted closes from 2000-06-06 to 2025-04-09
- **Federal Funds Rate**: Daily FEDFUNDS rate from FRED, aligned to trading calendar
- Forward-fill convention: ACT/360 day count

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data fetching and preprocessing
│   ├── ewma.py                 # EWMA volatility estimation
│   ├── algorithms.py           # Three volatility targeting algorithms
│   ├── metrics.py              # Performance metrics computation
│   ├── monte_carlo.py          # Monte Carlo confidence band generation
│   └── visualization.py        # Plotting functions
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_ewma.py
│   ├── test_algorithms.py
│   ├── test_metrics.py
│   └── test_monte_carlo.py
├── results/
│   ├── RESULTS.md              # Performance metrics table
│   ├── cumulative_returns.png
│   ├── running_volatility.png
│   └── monte_carlo_band.png
├── data/                       # Downloaded data cache
├── main.py                     # Main experiment runner
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full experiment:

```bash
python main.py
```

This will:
1. Download IVV and Fed Funds data
2. Run all four strategies
3. Compute performance metrics
4. Generate Monte Carlo confidence bands
5. Create visualizations
6. Save results to `results/` directory

## Testing

Run all tests:

```bash
pytest tests/ -v
```

## Results

Performance metrics and visualizations are saved in the `results/` directory:
- `RESULTS.md`: Comprehensive metrics table (Table 1 reproduction)
- `cumulative_returns.png`: Four-strategy cumulative return comparison
- `running_volatility.png`: Running annualized volatility with confidence bands
- `monte_carlo_band.png`: Monte Carlo confidence band visualization

## Key Parameters

- Target volatility: 15% annualized (σ_target = 0.15/√252)
- Leverage cap: 1.5
- EWMA halflife: 126 trading days (β = exp(-log(2)/126))
- Volatility controller gain: g = 50
- Leverage controller gain: g_ell = 20
- Control delay: 10 trading days
- Monte Carlo trials: 10,000
- Random seed: 42

## Requirements

- Python 3.8+
- numpy >= 1.26.4
- pandas >= 2.2.2
- matplotlib
- scipy >= 1.14.1
- yfinance (for data)
- pandas-datareader (for FRED data)

## License

MIT License

## References

This implementation follows the methodology described in the research paper on volatility targeting algorithms for passive index investing.
