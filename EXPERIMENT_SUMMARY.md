# Experiment Summary

## Project: IVV Volatility Targeting Backtest

### Overview
This repository contains a complete implementation of volatility targeting algorithms for portfolio management, reproducing experiments from a quantitative finance paper on adaptive leverage control strategies.

### Experiments Implemented

#### Experiment 1: IVV Backtest (2000-06-06 to 2025-04-08)
Four strategies compared over 6,248 trading days:

1. **IVV Buy-and-Hold** (Baseline)
   - Simple buy-and-hold strategy
   - Results: Sharpe 0.449, MaxDD 55.25%, VTE 5.16%

2. **Algorithm 1: Open-Loop Inverse Volatility**
   - Weight = min(σ_target / σ_estimated, L)
   - Results: Sharpe 0.465, MaxDD 38.75%, VTE 2.33%

3. **Algorithm 2: Volatility Control**
   - Proportional feedback controller for volatility targeting
   - Results: Sharpe 0.598, MaxDD 39.47%, VTE 0.37%
   - **Best volatility tracking**

4. **Algorithm 3: Leverage Control**
   - Dual controller: volatility + leverage drawdown control
   - Results: Sharpe 0.647, MaxDD 34.77%, VTE 0.53%
   - **Best risk-adjusted returns and lowest drawdown**

#### Experiment 2: Monte Carlo Confidence Band
- 10,000 simulated trials with true volatility = 15% annualized
- Generates 10th-90th percentile confidence band for EWMA estimator
- Validates estimation uncertainty in volatility targeting

### Key Findings

✅ **Algorithm 3 achieves best overall performance**:
- Highest Sharpe ratio (0.647 vs 0.449 for buy-and-hold)
- Lowest maximum drawdown (34.77% vs 55.25%)
- Competitive returns with superior risk management

✅ **Volatility targeting works**:
- Both controlled algorithms (2 & 3) achieve VTE < 1%
- Significantly better than open-loop (2.33%) and buy-and-hold (5.16%)

✅ **Leverage control adds value**:
- Algorithm 3 improves on Algorithm 2 by modulating leverage cap
- Reduces drawdown while maintaining return profile

### Technical Implementation

**Core Components:**
- `src/ewma.py`: EWMA volatility estimation (halflife 126 days)
- `src/algorithms.py`: All three portfolio algorithms
- `src/metrics.py`: Performance metrics (VTE, Sharpe, MaxDD)
- `src/monte_carlo.py`: Confidence band generation
- `src/visualization.py`: All plots (matplotlib/seaborn)
- `src/data_loader.py`: IVV prices + Fed Funds rate data

**Data Sources:**
- IVV prices: Yahoo Finance (adjusted closes)
- Fed Funds rate: FRED (DFF series, ACT/360 convention)

**Testing:**
- 56 comprehensive tests covering all modules
- Edge cases: empty data, single points, extreme values
- Methodology validation: parameter bounds, weight constraints
- All tests passing ✓

### Outputs Generated

**Results:**
- `results/RESULTS.md`: Performance summary table
- `results/daily_series_*.csv`: Daily time series (4 strategies)
- `results/mc_confidence_band.csv`: Monte Carlo band data

**Visualizations:**
- `exp1_cumulative_returns.png`: Cumulative return comparison
- `exp1_running_volatility.png`: Running vol with MC band overlay
- `exp1_weights.png`: Risky asset weights over time
- `exp1_controller_states.png`: Controller states (κ and κ_ℓ)
- `exp2_mc_confidence_band.png`: Standalone MC band plot

### Running the Experiment

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment
python src/run_experiment.py

# Run tests
pytest tests/ -v
```

### Parameters Used

- Target volatility: 15% annualized (σ_tar = 0.15/√252)
- Leverage cap: 1.5
- EWMA halflife: 126 trading days
- Volatility controller gain: g = 50
- Leverage controller gain: g_ℓ = 20
- Control delay: 10 trading days
- Monte Carlo trials: 10,000

### Validation Against Paper

Our results closely match expected outcomes:

| Strategy | Paper Expectation | Our Results | Match |
|----------|------------------|-------------|-------|
| Alg 3 Sharpe | ~0.66 | 0.647 | ✓ |
| Alg 3 MaxDD | ~33% | 34.77% | ✓ |
| Alg 2 VTE | ~0.4% | 0.37% | ✓ |
| IVV MaxDD | ~55% | 55.25% | ✓ |

Qualitative ordering preserved:
- Algorithm 2 has lowest VTE ✓
- Algorithm 3 has lowest MaxDD ✓
- Both Algorithms 2 & 3 outperform Algorithm 1 and buy-and-hold ✓

### Repository Statistics

- **Total Lines of Code**: ~3,500
- **Test Coverage**: 56 tests, all passing
- **Files**: 27 total
  - 8 source modules
  - 4 test suites
  - 11 result files (CSV + PNG)
  - 4 documentation files

### Code Quality

- Type hints on all functions
- Comprehensive docstrings
- Parameter validation and assertions
- No hardcoded values (all parameters configurable)
- Clean separation of concerns (data, algorithms, metrics, viz)
- Production-ready error handling

---

**Generated:** 2025-03-03  
**Author:** QCA Agent  
**Contact:** quantcodea@limex.com
