# IVV Backtest: Volatility Targeting Algorithms

This repository implements a comprehensive backtest comparing volatility targeting algorithms applied to the iShares Core S&P 500 ETF (IVV) from June 6, 2000 to April 9, 2025.

## Overview

The project reproduces experiments from a research paper on volatility targeting strategies, comparing:

1. **IVV Buy-and-Hold**: Baseline benchmark
2. **Algorithm 1 (Open-Loop)**: Inverse-volatility weighting with leverage cap
3. **Algorithm 2 (Volatility Control)**: Proportional feedback controller for volatility targeting
4. **Algorithm 3 (Leverage Control)**: Combined volatility and leverage drawdown control

## Experiments

### Experiment 1: IVV Backtest
Full backtest (June 6, 2000 - April 9, 2025) comparing all four strategies:
- Performance metrics: Volatility tracking error, annualized return, volatility, Sharpe ratio, max drawdown
- Cumulative return time series
- Running annualized volatility time series

### Experiment 2: Monte Carlo Confidence Band
Monte Carlo analysis to characterize the statistical uncertainty of the EWMA volatility estimator:
- 10th-90th percentile confidence band
- Overlay on empirical running volatility plot
- 10,000 trials with true volatility = 15% annualized

## Project Structure

```
.
├── src/
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── ewma.py              # EWMA volatility and MA functions
│   ├── algorithms.py        # Implementation of Algorithms 1, 2, 3
│   ├── metrics.py           # Performance metrics calculation
│   ├── monte_carlo.py       # Monte Carlo confidence band generation
│   ├── visualization.py     # Plotting functions
│   └── run_experiment.py    # Main experiment runner
├── tests/
│   └── test_*.py            # Comprehensive test suite
├── results/
│   ├── RESULTS.md           # Performance metrics and findings
│   ├── *.png                # Generated plots
│   └── *.csv                # Daily series data
├── data/                    # Downloaded market data (cached)
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Full Experiment

```bash
python src/run_experiment.py
```

This will:
1. Download IVV and Fed Funds rate data
2. Run all four strategies
3. Compute performance metrics
4. Generate plots
5. Save results to `results/` directory

### Run Tests

```bash
pytest tests/ -v
```

## Data Sources

- **IVV Daily Prices**: Yahoo Finance (adjusted close, accounts for splits and dividends)
- **Federal Funds Effective Rate**: FRED (DFF series, daily)
- **Trading Calendar**: Derived from IVV trading dates

## Methodology

### Parameters
- Target volatility: 15% annualized (σ_tar = 0.15/√252 daily)
- Leverage cap: 1.5
- EWMA halflife: 126 trading days (β = exp(-log(2)/126))
- Volatility controller gain: g = 50
- Leverage controller gain: g_ℓ = 20
- Control delay: 10 trading days
- Long MA halflife: 126 days
- Short MA halflife: 42 days

### EWMA Formula
Normalized EWMA variance estimator:
```
σ̂²_k = [(1-β)/(1-β^k)] * Σ_{j=1}^k β^(k-j) * x_j²
```

Implemented recursively:
```
S_k = (1-β) * x_k² + β * S_{k-1}
σ̂_k = √(S_k / (1-β^k))
```

### Algorithm Descriptions

**Algorithm 1 (Open-Loop)**:
```
w_k = min(σ_tar / σ̂_k, L)
```

**Algorithm 2 (Volatility Control)**:
```
e_k = log(σ̂^ind_k / σ_tar)
κ_k = (1-θ) * clip(-g*e_k, [κ_min, κ_max]) + θ*κ_{k-1}
w_k = min(exp(κ_k) * σ_tar / σ̂_k, L)
```

**Algorithm 3 (Leverage Control)**:
```
κ_ℓ,k = clip(-g_ℓ * log(MA^long_k / MA^short_k), [κ_ℓ,min, 0])
L_eff = exp(κ_ℓ,k) * L
w_k = min(exp(κ_k) * σ_tar / σ̂_k, L_eff)
```

## Results

Results are saved in `results/RESULTS.md` and include:
- Performance summary table (Table 1)
- Cumulative return plot
- Running annualized volatility plot with Monte Carlo confidence band

## License

MIT License

## References

This implementation is based on the research paper on volatility targeting algorithms for index investing.
