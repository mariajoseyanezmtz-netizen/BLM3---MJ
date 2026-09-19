# Black-Litterman Portfolio Optimization

Python implementation of a Black-Litterman portfolio optimization model using market data from Yahoo Finance.

## Universe

### Optimizable assets
- AAPL
- MSFT
- NVDA
- AMZN
- SPY
- QQQ

### Benchmark
- `^GSPC` (S&P 500)

The S&P 500 index is used only as a benchmark and is not included in the optimization.

## Methodology

The model follows this workflow:

1. Download historical adjusted prices from Yahoo Finance.
2. Calculate daily simple returns.
3. Estimate the annualized covariance matrix using Ledoit-Wolf shrinkage.
4. Construct the market portfolio using one of:
   - Market capitalization
   - Equal weight
   - SPY proxy
5. Obtain the risk-free rate manually or from Yahoo Finance.
6. Estimate risk aversion using either a fixed value or the market-implied formula.
7. Calculate equilibrium returns:

\[
\Pi = \delta \Sigma w_{mkt}
\]

8. Define investor views directly in `main.py`.
9. Construct the P and Q matrices for absolute and relative views.
10. Estimate `tau` statistically using bootstrap uncertainty of the equilibrium return vector.
11. Calculate He-Litterman Omega:

\[
\Omega = \tau P\Sigma P^T
\]

12. Calculate Black-Litterman posterior expected returns:

\[
\mu_{BL}
=
[(\tau\Sigma)^{-1}+P^T\Omega^{-1}P]^{-1}
[(\tau\Sigma)^{-1}\Pi+P^T\Omega^{-1}Q]
\]

13. Optimize the portfolio using:
   - Maximum Sharpe
   - Minimum volatility
   - Maximum mean-variance utility
14. Calculate risk metrics.
15. Construct and display the efficient frontier.
16. Compare the resulting portfolio with the S&P 500.

## Configuration

All model settings are in `main.py`.

Examples:

```python
LOOKBACK_YEARS = 5

MARKET_PORTFOLIO_METHOD = "market_cap"
# "market_cap", "equal_weight", "spy"

RISK_FREE_METHOD = "yahoo"
# "manual", "yahoo"

RISK_AVERSION_METHOD = "market_implied"
# "fixed", "market_implied"

OPTIMIZATION_OBJECTIVE = "max_sharpe"
# "max_sharpe", "min_volatility", "max_utility"

ALLOW_SHORT = False
MIN_WEIGHT = 0.00
MAX_WEIGHT = 0.30
```

## Investor Views

Views are defined **only in Python**, as requested.

Absolute view:

```python
VIEWS = [
    {"type": "absolute", "asset": "NVDA", "view": 0.15},
]
```

This means:

> Expected annual return of NVDA = 15%.

Relative view:

```python
VIEWS = [
    {
        "type": "relative",
        "asset_1": "NVDA",
        "asset_2": "MSFT",
        "view": 0.03,
    },
]
```

This means:

> NVDA is expected to outperform MSFT by 3 percentage points annually.

Multiple views can be combined.

If no views are supplied:

```python
REQUIRE_VIEWS = False
VIEWS = []
```

the model uses the equilibrium prior as the expected-return vector.

## Risk metrics

Historical 95% VaR:

\[
VaR_{95\%}=-Q_{0.05}(R_p)
\]

Historical 95% CVaR:

\[
CVaR_{95\%}
=
-E[R_p|R_p\le Q_{0.05}(R_p)]
\]

Maximum drawdown is also calculated.

## Important methodological note about tau

`tau` is estimated using a bootstrap procedure that measures the sampling uncertainty of the equilibrium return vector.

This is a modeling choice rather than a universal Black-Litterman identity. The procedure is explicitly implemented in `main.py` so that it can be reviewed and modified.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The script prints the model results to the console and displays the charts.

## Main outputs

- Market weights
- Historical annualized returns
- Ledoit-Wolf covariance matrix
- Equilibrium returns
- Investor views
- P and Q matrices
- Tau
- Omega
- Black-Litterman posterior returns
- Optimal portfolio weights
- Expected return
- Volatility
- Sharpe ratio
- Cumulative return
- Maximum drawdown
- Historical VaR
- Historical CVaR
- S&P 500 comparison
- Efficient frontier
- Portfolio charts

## Dependencies

The project intentionally does not use PyPortfolioOpt. Black-Litterman equations and portfolio optimization are implemented directly using NumPy and SciPy.

Ledoit-Wolf covariance estimation is implemented through scikit-learn's `LedoitWolf` estimator.
