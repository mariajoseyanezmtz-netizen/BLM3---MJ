
"""
Black-Litterman Portfolio Optimization
======================================
Yahoo Finance data | AAPL, MSFT, NVDA, AMZN, SPY, QQQ
Benchmark: ^GSPC

This file contains the quantitative model. Streamlit presentation is in
streamlit_app.py.
"""

from datetime import datetime, timedelta
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

ASSETS = ["AAPL", "MSFT", "NVDA", "AMZN", "SPY", "QQQ"]
BENCHMARK = "^GSPC"
TRADING_DAYS = 252

LOOKBACK_YEARS = 5
MARKET_PORTFOLIO_METHOD = "market_cap"  # market_cap | equal_weight | spy

RISK_FREE_METHOD = "yahoo"               # yahoo | manual
MANUAL_RISK_FREE_RATE = 0.04
RISK_FREE_TICKER = "^IRX"

RISK_AVERSION_METHOD = "market_implied"  # market_implied | fixed
FIXED_RISK_AVERSION = 2.5

# Views ONLY in Python.
VIEWS = [
    # {"type": "absolute", "asset": "NVDA", "view": 0.15},
    # {"type": "relative", "asset_1": "NVDA", "asset_2": "MSFT", "view": 0.03},
]

TAU_METHOD = "statistical"
BOOTSTRAP_SAMPLES = 300
RANDOM_SEED = 42

OMEGA_METHOD = "he_litterman"

OPTIMIZATION_OBJECTIVE = "max_sharpe"  # max_sharpe | min_volatility | max_utility
UTILITY_LAMBDA = 2.5

ALLOW_SHORT = False
MIN_WEIGHT = 0.00
MAX_WEIGHT = 0.30

N_EFFICIENT_PORTFOLIOS = 1000
VAR_CONFIDENCE = 0.95

# If Yahoo blocks quote metadata in a cloud runtime, the model can still
# run with an explicit documented fallback. This avoids silently inventing
# market caps.
MARKET_CAP_FALLBACK = "equal_weight"


def download_prices(tickers, years):
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25 + 10))
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no historical price data.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Adj Close" in level0:
            prices = raw["Adj Close"].copy()
        elif "Close" in level0:
            prices = raw["Close"].copy()
        else:
            raise RuntimeError("Yahoo Finance response has no Adj Close/Close field.")
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[field]].copy()
        prices.columns = [tickers[0]]

    prices = prices.reindex(columns=tickers).dropna(how="any")
    if prices.empty:
        raise RuntimeError("No common observations were returned by Yahoo Finance.")
    return prices


def calculate_returns(prices):
    return prices.pct_change().dropna(how="any")


def ledoit_wolf_covariance(returns):
    model = LedoitWolf().fit(returns.values)
    cov = model.covariance_ * TRADING_DAYS
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns), model.shrinkage_


def annualized_mean(returns):
    return returns.mean() * TRADING_DAYS


def _valid_number(x):
    try:
        x = float(x)
        return np.isfinite(x) and x > 0
    except Exception:
        return False


def get_market_caps(tickers, reference_prices):
    """
    Obtain current market caps using Yahoo Finance only.

    yfinance's FastInfo can derive market cap from shares and latest price.
    We try FastInfo, Yahoo valuation measures, Yahoo info, and Yahoo shares.
    If Yahoo quote metadata is unavailable in the cloud, we use the explicit
    configured fallback instead of crashing or fabricating values.
    """
    caps = {}
    diagnostics = {}

    for ticker in tickers:
        cap = np.nan
        method = "unavailable"

        try:
            t = yf.Ticker(ticker)

            # 1. FastInfo: preferred path. yfinance documents fast_info and
            # its implementation can calculate market cap from shares*price.
            try:
                fi = t.fast_info
                try:
                    cap = fi["market_cap"]
                except Exception:
                    cap = getattr(fi, "market_cap", np.nan)
                if _valid_number(cap):
                    method = "Yahoo fast_info"
            except Exception as e:
                diagnostics[f"{ticker}:fast_info"] = type(e).__name__

            # 2. Yahoo valuation time series.
            if not _valid_number(cap):
                try:
                    v = t.get_valuation_measures(freq="trailing", periods=0)
                    if "Market Cap" in v.index and "Current" in v.columns:
                        cap = v.loc["Market Cap", "Current"]
                    elif "MarketCap" in v.index and "Current" in v.columns:
                        cap = v.loc["MarketCap", "Current"]
                    if _valid_number(cap):
                        method = "Yahoo valuation measures"
                except Exception as e:
                    diagnostics[f"{ticker}:valuation"] = type(e).__name__

            # 3. Yahoo quote info.
            if not _valid_number(cap):
                try:
                    info = t.get_info()
                    cap = info.get("marketCap", np.nan)
                    if _valid_number(cap):
                        method = "Yahoo info"
                except Exception as e:
                    diagnostics[f"{ticker}:info"] = type(e).__name__

            # 4. Yahoo shares outstanding * Yahoo historical latest price.
            if not _valid_number(cap):
                try:
                    shares = t.get_shares_full(
                        start=datetime.now() - timedelta(days=120),
                        end=datetime.now()
                    )
                    if shares is not None and len(shares):
                        latest_shares = float(pd.Series(shares).dropna().iloc[-1])
                        px = float(reference_prices.loc[ticker])
                        candidate = latest_shares * px
                        if _valid_number(candidate):
                            cap = candidate
                            method = "Yahoo shares × Yahoo price"
                except Exception as e:
                    diagnostics[f"{ticker}:shares"] = type(e).__name__

        except Exception as e:
            diagnostics[f"{ticker}:ticker"] = type(e).__name__

        if _valid_number(cap):
            caps[ticker] = float(cap)
        else:
            caps[ticker] = np.nan

        diagnostics[f"{ticker}:method"] = method

    caps = pd.Series(caps, dtype=float)

    if caps.notna().all() and (caps > 0).all():
        return caps, diagnostics, False

    # Explicit fallback. This is not disguised as market-cap weighting.
    if MARKET_CAP_FALLBACK == "equal_weight":
        return (
            pd.Series(1.0, index=tickers, dtype=float),
            diagnostics,
            True,
        )

    missing = caps[caps.isna() | (caps <= 0)].index.tolist()
    raise RuntimeError(
        f"Yahoo Finance did not expose usable market-cap metadata for: {missing}. "
        "Set MARKET_CAP_FALLBACK='equal_weight' or retry later."
    )


def market_weights(assets, returns, prices):
    method = MARKET_PORTFOLIO_METHOD.lower()

    if method == "equal_weight":
        return pd.Series(1 / len(assets), index=assets), {"method": "equal_weight"}

    if method == "spy":
        # SPY is a market proxy. We allocate among the selected universe
        # using positive beta-to-SPY weights.
        cov = returns[assets].cov() * TRADING_DAYS
        spy = returns["SPY"]
        var_spy = float(spy.var() * TRADING_DAYS)
        betas = (cov["SPY"] / var_spy).clip(lower=0)
        if betas.sum() <= 0:
            return pd.Series(1 / len(assets), index=assets), {"method": "spy -> equal-weight fallback"}
        return betas / betas.sum(), {"method": "spy beta proxy"}

    if method == "market_cap":
        caps, diagnostics, fallback = get_market_caps(
            assets, prices.iloc[-1].reindex(assets)
        )
        weights = caps / caps.sum()
        info = {
            "method": "Yahoo market-cap" if not fallback else "equal-weight fallback (Yahoo market-cap unavailable)",
            "market_caps": caps,
            "diagnostics": diagnostics,
        }
        return weights, info

    raise ValueError("MARKET_PORTFOLIO_METHOD must be market_cap, equal_weight, or spy.")


def get_risk_free_rate():
    if RISK_FREE_METHOD.lower() == "manual":
        return float(MANUAL_RISK_FREE_RATE)

    end = datetime.now()
    start = end - timedelta(days=45)
    raw = yf.download(
        RISK_FREE_TICKER,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance did not return ^IRX data.")

    if isinstance(raw.columns, pd.MultiIndex):
        series = raw["Close"].squeeze()
    else:
        series = raw["Close"]
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        raise RuntimeError("Yahoo Finance returned no valid ^IRX observations.")
    return float(series.iloc[-1]) / 100.0


def build_views(views, assets):
    if not views:
        return None, None
    rows, q = [], []
    for view in views:
        typ = view["type"].lower()
        row = np.zeros(len(assets))
        if typ == "absolute":
            row[assets.index(view["asset"])] = 1.0
        elif typ == "relative":
            row[assets.index(view["asset_1"])] = 1.0
            row[assets.index(view["asset_2"])] = -1.0
        else:
            raise ValueError("View type must be absolute or relative.")
        rows.append(row)
        q.append(float(view["view"]))
    return np.asarray(rows, dtype=float), np.asarray(q, dtype=float)


def estimate_tau_bootstrap(returns, market_weights_series, delta, cov_full):
    rng = np.random.default_rng(RANDOM_SEED)
    X = returns.values
    n = len(X)
    pis = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = X[rng.integers(0, n, size=n)]
        lw = LedoitWolf().fit(sample)
        sample_cov = lw.covariance_ * TRADING_DAYS
        pis.append(delta * sample_cov @ market_weights_series.values)
    boot_cov = np.cov(np.asarray(pis), rowvar=False, ddof=1)
    num = np.trace(boot_cov) / boot_cov.shape[0]
    den = np.trace(cov_full.values) / cov_full.shape[0]
    tau = num / den if den > 0 else np.nan
    if not np.isfinite(tau) or tau <= 0:
        tau = 1.0 / n
    return float(tau), boot_cov


def black_litterman_posterior(pi, cov, tau, P, Q):
    Sigma = cov.values
    tau_sigma = tau * Sigma
    omega = tau * P @ Sigma @ P.T
    omega = np.atleast_2d(omega) + np.eye(len(Q)) * 1e-12
    inv_tau = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(omega)
    A = inv_tau + P.T @ inv_omega @ P
    b = inv_tau @ pi.values + P.T @ inv_omega @ Q
    posterior = np.linalg.solve(A, b)
    return pd.Series(posterior, index=cov.index), omega


def portfolio_return(w, mu):
    return float(w @ mu)


def portfolio_volatility(w, cov):
    return float(np.sqrt(max(w @ cov @ w, 0)))


def optimize_portfolio(expected_returns, cov, risk_free_rate):
    n = len(expected_returns)
    x0 = np.ones(n) / n
    bounds = [(None, None)] * n if ALLOW_SHORT else [(MIN_WEIGHT, MAX_WEIGHT)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def obj(w):
        ret = w @ expected_returns.values
        vol = portfolio_volatility(w, cov.values)
        if OPTIMIZATION_OBJECTIVE == "max_sharpe":
            return -(ret - risk_free_rate) / vol if vol > 0 else 1e9
        if OPTIMIZATION_OBJECTIVE == "min_volatility":
            return vol
        if OPTIMIZATION_OBJECTIVE == "max_utility":
            return -(ret - 0.5 * UTILITY_LAMBDA * vol**2)
        raise ValueError("Invalid OPTIMIZATION_OBJECTIVE.")

    result = minimize(
        obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-10}
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return pd.Series(result.x, index=expected_returns.index)


def min_variance_for_target(target, mu, cov):
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(None, None)] * n if ALLOW_SHORT else [(MIN_WEIGHT, MAX_WEIGHT)] * n
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ mu.values - target},
    ]
    r = minimize(
        lambda w: portfolio_volatility(w, cov.values),
        x0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-10}
    )
    if not r.success:
        return None
    return r.x @ mu.values, portfolio_volatility(r.x, cov.values)


def efficient_frontier(mu, cov, n_points):
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(None, None)] * n if ALLOW_SHORT else [(MIN_WEIGHT, MAX_WEIGHT)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    lo = minimize(lambda w: w @ mu.values, x0, method="SLSQP", bounds=bounds, constraints=cons)
    hi = minimize(lambda w: -(w @ mu.values), x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not lo.success or not hi.success:
        raise RuntimeError("Could not determine feasible efficient-frontier range.")
    targets = np.linspace(lo.x @ mu.values, hi.x @ mu.values, n_points)
    vols, rets = [], []
    for target in targets:
        p = min_variance_for_target(target, mu, cov)
        if p is not None:
            rets.append(p[0]); vols.append(p[1])
    return np.asarray(vols), np.asarray(rets)


def historical_var_cvar(daily_returns, confidence=0.95):
    cutoff = daily_returns.quantile(1 - confidence)
    var = -float(cutoff)
    tail = daily_returns[daily_returns <= cutoff]
    cvar = -float(tail.mean()) if len(tail) else np.nan
    return var, cvar


def max_drawdown(daily_returns):
    wealth = (1 + daily_returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min()), dd


def annualized_metrics(daily_returns, rf):
    ret = float(daily_returns.mean() * TRADING_DAYS)
    vol = float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "annual_return": ret,
        "annual_volatility": vol,
        "sharpe": (ret - rf) / vol if vol > 0 else np.nan,
        "cumulative_return": float((1 + daily_returns).prod() - 1),
    }


def run_model(lookback_years=None):
    years = LOOKBACK_YEARS if lookback_years is None else int(lookback_years)
    if years not in (1, 3, 5):
        raise ValueError("lookback_years must be 1, 3, or 5.")

    prices = download_prices(ASSETS + [BENCHMARK], years)
    asset_prices = prices[ASSETS]
    benchmark_prices = prices[BENCHMARK]
    asset_returns = calculate_returns(asset_prices)
    benchmark_returns = calculate_returns(benchmark_prices)
    benchmark_returns = benchmark_returns.reindex(asset_returns.index).dropna()
    asset_returns = asset_returns.reindex(benchmark_returns.index).dropna()

    cov, shrinkage = ledoit_wolf_covariance(asset_returns)
    hist_mean = annualized_mean(asset_returns)
    mkt_w, mkt_info = market_weights(ASSETS, asset_returns, asset_prices)

    rf = get_risk_free_rate()
    market_return = float(mkt_w.values @ hist_mean.values)
    market_variance = float(mkt_w.values @ cov.values @ mkt_w.values)

    if RISK_AVERSION_METHOD == "fixed":
        delta = FIXED_RISK_AVERSION
    else:
        delta = (market_return - rf) / market_variance
        if not np.isfinite(delta) or delta <= 0:
            delta = FIXED_RISK_AVERSION

    pi = pd.Series(delta * cov.values @ mkt_w.values, index=ASSETS)

    P, Q = build_views(VIEWS, ASSETS)
    if P is None:
        tau = np.nan
        omega = None
        posterior = pi.copy()
    else:
        tau, _ = estimate_tau_bootstrap(asset_returns, mkt_w, delta, cov)
        posterior, omega = black_litterman_posterior(pi, cov, tau, P, Q)

    weights = optimize_portfolio(posterior, cov, rf)
    frontier_vol, frontier_ret = efficient_frontier(posterior, cov, N_EFFICIENT_PORTFOLIOS)

    portfolio_daily = asset_returns @ weights
    metrics = annualized_metrics(portfolio_daily, rf)
    metrics["max_drawdown"] = max_drawdown(portfolio_daily)[0]
    metrics["var_95"], metrics["cvar_95"] = historical_var_cvar(portfolio_daily, VAR_CONFIDENCE)
    benchmark_metrics = annualized_metrics(benchmark_returns, rf)
    benchmark_metrics["max_drawdown"] = max_drawdown(benchmark_returns)[0]

    return {
        "prices": asset_prices,
        "returns": asset_returns,
        "benchmark_returns": benchmark_returns,
        "covariance": cov,
        "shrinkage": shrinkage,
        "historical_mean": hist_mean,
        "market_weights": mkt_w,
        "market_info": mkt_info,
        "risk_free_rate": rf,
        "delta": delta,
        "pi": pi,
        "P": P,
        "Q": Q,
        "tau": tau,
        "omega": omega,
        "posterior": posterior,
        "weights": weights,
        "frontier_vol": frontier_vol,
        "frontier_ret": frontier_ret,
        "portfolio_daily": portfolio_daily,
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
    }


if __name__ == "__main__":
    result = run_model()
    print("\nBLACK-LITTERMAN PORTFOLIO")
    print(result["weights"].to_string(float_format=lambda x: f"{x:.2%}"))
    print("\nPOSTERIOR RETURNS")
    print(result["posterior"].to_string(float_format=lambda x: f"{x:.2%}"))
