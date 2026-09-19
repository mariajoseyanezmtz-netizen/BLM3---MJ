# Black-Litterman Portfolio Optimization

Academic Black-Litterman portfolio optimization model using Yahoo Finance.

## Assets

AAPL, MSFT, NVDA, AMZN, SPY, QQQ

Benchmark: `^GSPC`

## Files

- `main.py` — quantitative model and Black-Litterman calculations.
- `streamlit_app.py` — Streamlit interface.
- `requirements.txt` — dependencies.
- `.gitignore` — Git exclusions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Model

1. Yahoo Finance adjusted historical prices.
2. Daily simple returns.
3. Ledoit-Wolf annualized covariance.
4. Market portfolio.
5. Risk-free rate from Yahoo Finance (`^IRX`).
6. Market-implied risk aversion.
7. Equilibrium returns:

\[
\Pi = \delta\Sigma w_{mkt}
\]

8. Investor views entered directly in Python.
9. He-Litterman uncertainty:

\[
\Omega = \tau P\Sigma P^T
\]

10. Posterior:

\[
\mu_{BL} =
[(\tau\Sigma)^{-1}+P^T\Omega^{-1}P]^{-1}
[(\tau\Sigma)^{-1}\Pi+P^T\Omega^{-1}Q]
\]

11. Portfolio optimization.
12. Efficient frontier.
13. Historical VaR/CVaR and maximum drawdown.
14. Comparison with `^GSPC`.

## Yahoo Finance market-cap handling

The model first attempts Yahoo `fast_info`, then Yahoo valuation measures, Yahoo `info`, and Yahoo shares outstanding multiplied by the Yahoo price. These are all Yahoo Finance-backed methods. `yfinance` documents `fast_info`, `get_info`, `get_valuation_measures`, and `get_shares_full`.

If a cloud runtime blocks Yahoo quote metadata, the code uses an explicit **equal-weight fallback** rather than fabricating market-cap values or crashing. The Streamlit interface clearly warns when this happens.

## Investor views

Absolute:

```python
VIEWS = [
    {"type": "absolute", "asset": "NVDA", "view": 0.15},
]
```

Relative:

```python
VIEWS = [
    {"type": "relative", "asset_1": "NVDA", "asset_2": "MSFT", "view": 0.03},
]
```

Views are defined only in Python; no CSV file is required.
