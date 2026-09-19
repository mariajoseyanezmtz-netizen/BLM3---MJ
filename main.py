"""
Black-Litterman Portfolio Optimization
======================================

Universe:
    AAPL, MSFT, NVDA, AMZN, SPY, QQQ
Benchmark:
    ^GSPC (S&P 500)

Data source:
    Yahoo Finance via yfinance

The model is implemented directly with NumPy/SciPy/scikit-learn.
PyPortfolioOpt is intentionally not used.

Run:
    pip install -r requirements.txt
    python main.py
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

ASSETS = ["AAPL", "MSFT", "NVDA", "AMZN", "SPY", "QQQ"]
BENCHMARK = "^GSPC"

LOOKBACK_YEARS = 5          # Allowed: 1, 3, 5
TRADING_DAYS = 252

# Market portfolio:
# "market_cap", "equal_weight", "spy"
MARKET_PORTFOLIO_METHOD = "market_cap"

# Risk-free rate:
# "manual" or "yahoo"
RISK_FREE_METHOD = "yahoo"
MANUAL_RISK_FREE_RATE = 0.04
RISK_FREE_TICKER = "^IRX"   # 13-week Treasury Bill yield proxy

# Risk aversion:
# "fixed" or "market_implied"
RISK_AVERSION_METHOD = "market_implied"
FIXED_RISK_AVERSION = 2.5

# Views are optional.
REQUIRE_VIEWS = False

# Views are ONLY defined here in Python.
# Absolute example:
# {"type": "absolute", "asset": "NVDA", "view": 0.15}
#
# Relative example:
# {"type": "relative", "asset_1": "NVDA",
#  "asset_2": "MSFT", "view": 0.03}
VIEWS = [
    # Uncomment/edit examples as desired:
    # {"type": "absolute", "asset": "NVDA", "view": 0.15},
    # {"type": "relative", "asset_1": "NVDA", "asset_2": "MSFT", "view": 0.03},
]

# Omega:
# He-Litterman
OMEGA_METHOD = "he_litterman"

# Tau:
# "statistical" = bootstrap-based prior uncertainty calibration
TAU_METHOD = "statistical"
BOOTSTRAP_SAMPLES = 300
RANDOM_SEED = 42

# Optimization:
# "max_sharpe", "min_volatility", "max_utility"
OPTIMIZATION_OBJECTIVE = "max_sharpe"

# Position constraints
ALLOW_SHORT = False
MIN_WEIGHT = 0.00
MAX_WEIGHT = 0.30

# Mean-variance utility risk-aversion parameter
UTILITY_LAMBDA = 2.5

# Efficient frontier
N_EFFICIENT_PORTFOLIOS = 1000

# Historical VaR/CVaR
VAR_CONFIDENCE = 0.95

# ============================================================
# DATA
# ============================================================

def download_prices(tickers, years):
    """Download adjusted historical prices from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25 + 10))

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    # yfinance can return MultiIndex columns for multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            raise RuntimeError("Could not find Adj Close/Close in Yahoo Finance data.")
    else:
        # Single-ticker fallback
        if "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy()
            prices.columns = [tickers[0]]
        elif "Close" in raw.columns:
            prices = raw[["Close"]].copy()
            prices.columns = [tickers[0]]
        else:
            raise RuntimeError("Could not find Adj Close/Close in Yahoo Finance data.")

    prices = prices.reindex(columns=tickers)
    prices = prices.dropna(how="any")
    return prices


def calculate_returns(prices):
    """Simple daily returns; incomplete observations are removed."""
    return prices.pct_change().dropna(how="any")


# ============================================================
# COVARIANCE / STATISTICS
# ============================================================

def ledoit_wolf_covariance(returns):
    """Annualized Ledoit-Wolf covariance matrix."""
    model = LedoitWolf().fit(returns.values)
    daily_cov = model.covariance_
    annual_cov = daily_cov * TRADING_DAYS
    return pd.DataFrame(
        annual_cov, index=returns.columns, columns=returns.columns
    ), model.shrinkage_


def annualized_mean(returns):
    return returns.mean() * TRADING_DAYS


# ============================================================
# MARKET PORTFOLIO
# ============================================================

def get_market_caps(tickers):
    """Get current market capitalization from Yahoo Finance."""
    caps = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            cap = info.get("market_cap", np.nan)
            if pd.isna(cap):
                info2 = yf.Ticker(ticker).info
                cap = info2.get("marketCap", np.nan)
            caps[ticker] = float(cap)
        except Exception:
            caps[ticker] = np.nan

    caps = pd.Series(caps, dtype=float)

    if caps.isna().any() or (caps <= 0).any():
        missing = caps[caps.isna() | (caps <= 0)].index.tolist()
        raise RuntimeError(
            f"Could not obtain valid market caps from Yahoo Finance for: {missing}"
        )

    return caps


def market_weights(assets, returns, prices):
    """Build the chosen market portfolio."""
    method = MARKET_PORTFOLIO_METHOD.lower()

    if method == "equal_weight":
        weights = pd.Series(1.0 / len(assets), index=assets)

    elif method == "spy":
        # SPY is used as the market proxy. Its return stream is translated
        # into implied relative weights using the asset beta structure.
        # The resulting positive weights are normalized.
        cov = returns[assets].cov() * TRADING_DAYS
        spy_returns = returns["SPY"] if "SPY" in returns.columns else returns["SPY"]
        var_spy = spy_returns.var() * TRADING_DAYS
        betas = cov["SPY"] / var_spy
        betas = betas.clip(lower=0)
        if betas.sum() <= 0:
            weights = pd.Series(1.0 / len(assets), index=assets)
        else:
            weights = betas / betas.sum()

    elif method == "market_cap":
        caps = get_market_caps(assets)
        weights = caps / caps.sum()

    else:
        raise ValueError(
            "MARKET_PORTFOLIO_METHOD must be 'market_cap', 'equal_weight', or 'spy'."
        )

    return weights


# ============================================================
# RISK-FREE RATE
# ============================================================

def get_risk_free_rate():
    if RISK_FREE_METHOD.lower() == "manual":
        return float(MANUAL_RISK_FREE_RATE)

    if RISK_FREE_METHOD.lower() != "yahoo":
        raise ValueError("RISK_FREE_METHOD must be 'manual' or 'yahoo'.")

    # ^IRX is quoted by Yahoo as a percentage yield.
    end = datetime.now()
    start = end - timedelta(days=30)

    raw = yf.download(
        RISK_FREE_TICKER,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError("Could not download the Yahoo Finance risk-free proxy.")

    if isinstance(raw.columns, pd.MultiIndex):
        series = raw["Close"].squeeze()
    else:
        series = raw["Close"]

    series = series.dropna()

    if series.empty:
        raise RuntimeError("Risk-free proxy returned no valid observations.")

    # ^IRX is quoted as percent, e.g. 4.50 means 4.50%.
    return float(series.iloc[-1]) / 100.0


# ============================================================
# BLACK-LITTERMAN INPUTS
# ============================================================

def market_implied_risk_aversion(market_weights_series, cov, market_return):
    market_variance = float(
        market_weights_series.values @ cov.values @ market_weights_series.values
    )
    if market_variance <= 0:
        raise RuntimeError("Market variance must be positive.")

    return (market_return - RISK_FREE_RATE) / market_variance


def build_views(views, assets):
    """
    Convert Python view dictionaries into P and Q.

    Absolute:
        {"type": "absolute", "asset": "NVDA", "view": 0.15}

    Relative:
        {"type": "relative", "asset_1": "NVDA",
         "asset_2": "MSFT", "view": 0.03}
    """
    if not views:
        return None, None

    P_rows = []
    Q_values = []

    for view in views:
        vtype = view["type"].lower()

        if vtype == "absolute":
            asset = view["asset"]
            q = float(view["view"])

            row = np.zeros(len(assets))
            row[assets.index(asset)] = 1.0

        elif vtype == "relative":
            asset_1 = view["asset_1"]
            asset_2 = view["asset_2"]
            q = float(view["view"])

            row = np.zeros(len(assets))
            row[assets.index(asset_1)] = 1.0
            row[assets.index(asset_2)] = -1.0

        else:
            raise ValueError("View type must be 'absolute' or 'relative'.")

        P_rows.append(row)
        Q_values.append(q)

    return np.asarray(P_rows, dtype=float), np.asarray(Q_values, dtype=float)


# ============================================================
# TAU: STATISTICAL / BOOTSTRAP ESTIMATION
# ============================================================

def estimate_tau_bootstrap(returns, market_weights_series, delta, cov_full):
    """
    Estimate tau from bootstrap uncertainty of the equilibrium return vector.

    For each bootstrap sample:
        1. estimate covariance with Ledoit-Wolf
        2. calculate market-implied equilibrium returns Pi_b
        3. collect Pi_b

    We then calibrate a scalar tau so that tau * Sigma has, on average,
    the same diagonal scale as the bootstrap covariance of Pi.

    tau = mean(diag(Cov_boot(Pi))) / mean(diag(Sigma))

    This is a modeling choice, not a universal Black-Litterman identity.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    X = returns.values
    n = len(X)

    pi_boot = np.empty((BOOTSTRAP_SAMPLES, len(returns.columns)))

    for b in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, size=n)
        sample = X[idx]

        lw = LedoitWolf().fit(sample)
        sample_cov = lw.covariance_ * TRADING_DAYS

        pi_b = delta * (sample_cov @ market_weights_series.values)
        pi_boot[b, :] = pi_b

    bootstrap_cov = np.cov(pi_boot, rowvar=False, ddof=1)

    numerator = np.trace(bootstrap_cov) / bootstrap_cov.shape[0]
    denominator = np.trace(cov_full.values) / cov_full.shape[0]

    tau = numerator / denominator if denominator > 0 else np.nan

    # Numerical guardrails: tau must be positive and finite.
    if not np.isfinite(tau) or tau <= 0:
        tau = 1.0 / n

    return float(tau), bootstrap_cov


# ============================================================
# BLACK-LITTERMAN POSTERIOR
# ============================================================

def black_litterman_posterior(pi, cov, tau, P, Q, omega):
    """
    Compute posterior Black-Litterman expected returns.

    mu_BL =
      inv(inv(tau*Sigma) + P' inv(Omega) P)
      @
      (inv(tau*Sigma) Pi + P' inv(Omega) Q)
    """
    Sigma = cov.values
    pi_vec = pi.values

    tau_sigma = tau * Sigma

    A = np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P
    b = np.linalg.inv(tau_sigma) @ pi_vec + P.T @ np.linalg.inv(omega) @ Q

    posterior = np.linalg.solve(A, b)

    return pd.Series(posterior, index=cov.index)


# ============================================================
# PORTFOLIO OPTIMIZATION
# ============================================================

def portfolio_return(weights, expected_returns):
    return float(weights @ expected_returns)


def portfolio_volatility(weights, cov):
    return float(np.sqrt(weights @ cov @ weights))


def objective_function(weights, expected_returns, cov, objective):
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov)

    if objective == "max_sharpe":
        if vol <= 0:
            return 1e6
        return -(ret - RISK_FREE_RATE) / vol

    if objective == "min_volatility":
        return vol

    if objective == "max_utility":
        return -(ret - 0.5 * UTILITY_LAMBDA * (vol ** 2))

    raise ValueError(
        "OPTIMIZATION_OBJECTIVE must be 'max_sharpe', "
        "'min_volatility', or 'max_utility'."
    )


def optimize_portfolio(expected_returns, cov):
    n = len(expected_returns)
    x0 = np.ones(n) / n

    bounds = (
        [(None, None)] * n
        if ALLOW_SHORT
        else [(MIN_WEIGHT, MAX_WEIGHT)] * n
    )

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    }

    result = minimize(
        objective_function,
        x0,
        args=(expected_returns.values, cov.values, OPTIMIZATION_OBJECTIVE),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=expected_returns.index)


# ============================================================
# EFFICIENT FRONTIER
# ============================================================

def min_variance_for_target_return(target_return, expected_returns, cov):
    n = len(expected_returns)
    x0 = np.ones(n) / n

    bounds = (
        [(None, None)] * n
        if ALLOW_SHORT
        else [(MIN_WEIGHT, MAX_WEIGHT)] * n
    )

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {
            "type": "eq",
            "fun": lambda w: np.dot(w, expected_returns.values) - target_return
        },
    ]

    result = minimize(
        lambda w: portfolio_volatility(w, cov.values),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10, "disp": False},
    )

    if not result.success:
        return None

    w = result.x
    return portfolio_return(w, expected_returns.values), portfolio_volatility(
        w, cov.values
    )


def efficient_frontier(expected_returns, cov, n_points):
    # Determine feasible endpoint returns from individual optimization.
    n = len(expected_returns)
    bounds = (
        [(None, None)] * n
        if ALLOW_SHORT
        else [(MIN_WEIGHT, MAX_WEIGHT)] * n
    )

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    x0 = np.ones(n) / n

    res_min = minimize(
        lambda w: np.dot(w, expected_returns.values),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    res_max = minimize(
        lambda w: -np.dot(w, expected_returns.values),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not res_min.success or not res_max.success:
        raise RuntimeError("Could not determine feasible return range.")

    r_min = portfolio_return(res_min.x, expected_returns.values)
    r_max = portfolio_return(res_max.x, expected_returns.values)

    targets = np.linspace(r_min, r_max, n_points)

    vols = []
    rets = []

    for target in targets:
        point = min_variance_for_target_return(target, expected_returns, cov)
        if point is not None:
            rets.append(point[0])
            vols.append(point[1])

    return np.asarray(vols), np.asarray(rets)


# ============================================================
# RISK METRICS
# ============================================================

def portfolio_daily_returns(weights, returns):
    return returns @ weights.values


def max_drawdown(daily_returns):
    wealth = (1.0 + daily_returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min()), drawdown


def historical_var_cvar(daily_returns, confidence=0.95):
    alpha = 1.0 - confidence
    cutoff = daily_returns.quantile(alpha)

    var = -float(cutoff)

    tail = daily_returns[daily_returns <= cutoff]
    cvar = -float(tail.mean()) if len(tail) > 0 else np.nan

    return var, cvar


def annualized_metrics(daily_returns):
    annual_return = float(daily_returns.mean() * TRADING_DAYS)
    annual_vol = float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (
        (annual_return - RISK_FREE_RATE) / annual_vol
        if annual_vol > 0 else np.nan
    )

    cumulative = float((1.0 + daily_returns).prod() - 1.0)

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "cumulative_return": cumulative,
    }


# ============================================================
# OUTPUT
# ============================================================

def print_vector(title, series, pct=True):
    print("\n" + title)
    print("-" * len(title))
    for name, value in series.items():
        if pct:
            print(f"{name:<10} {value:>10.2%}")
        else:
            print(f"{name:<10} {value:>10.6f}")


def print_matrix(title, matrix):
    print("\n" + title)
    print("-" * len(title))
    print(matrix.to_string(float_format=lambda x: f"{x: .6f}"))


# ============================================================
# PLOTS
# ============================================================

def plot_results(prices, returns, weights, pi, posterior, frontier_vol, frontier_ret,
                 benchmark_returns, benchmark_name):
    sns.set_theme(style="whitegrid")

    # 1. Normalized price evolution
    normalized = prices / prices.iloc[0]

    plt.figure(figsize=(12, 6))
    for col in normalized.columns:
        plt.plot(normalized.index, normalized[col], label=col)
    plt.title("Normalized Price Evolution")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Correlation matrix
    plt.figure(figsize=(9, 7))
    sns.heatmap(returns.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Daily Return Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # 3. Portfolio weights
    plt.figure(figsize=(10, 6))
    weights.plot(kind="bar")
    plt.title("Optimal Portfolio Weights")
    plt.ylabel("Weight")
    plt.axhline(0, linewidth=0.8)
    plt.tight_layout()
    plt.show()

    # 4. Prior vs posterior
    comparison = pd.DataFrame({"Prior": pi, "Black-Litterman": posterior})

    plt.figure(figsize=(11, 6))
    comparison.plot(kind="bar", ax=plt.gca())
    plt.title("Equilibrium Prior vs. Black-Litterman Returns")
    plt.ylabel("Expected Annual Return")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 5. Efficient frontier
    plt.figure(figsize=(10, 7))
    plt.scatter(frontier_vol, frontier_ret, s=8, alpha=0.6, label="Efficient Frontier")

    selected_ret = float(weights.values @ posterior.values)
    selected_vol = float(np.sqrt(weights.values @
                                  np.cov(returns.values, rowvar=False) * TRADING_DAYS
                                  @ weights.values))

    # Use model covariance for the selected portfolio volatility.
    plt.scatter(selected_vol, selected_ret, s=120, marker="*", label="Selected BL Portfolio")

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Expected Annual Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. Cumulative portfolio vs benchmark
    portfolio_daily = returns @ weights
    common = pd.concat(
        [portfolio_daily.rename("Black-Litterman"), benchmark_returns.rename(benchmark_name)],
        axis=1
    ).dropna()

    cumulative = (1 + common).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative.index, cumulative["Black-Litterman"], label="Black-Litterman")
    plt.plot(cumulative.index, cumulative[benchmark_name], label=benchmark_name)
    plt.title("Cumulative Performance vs. S&P 500")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. Drawdown
    portfolio_dd, portfolio_drawdown = max_drawdown(portfolio_daily)
    benchmark_dd, benchmark_drawdown = max_drawdown(benchmark_returns.dropna())

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_drawdown.index, portfolio_drawdown, label="Black-Litterman")
    plt.plot(benchmark_drawdown.index, benchmark_drawdown, label=benchmark_name)
    plt.title("Drawdown")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    global RISK_FREE_RATE

    if LOOKBACK_YEARS not in [1, 3, 5]:
        raise ValueError("LOOKBACK_YEARS must be 1, 3, or 5.")

    print("=" * 72)
    print("BLACK-LITTERMAN PORTFOLIO OPTIMIZATION")
    print("=" * 72)

    tickers = ASSETS + [BENCHMARK]

    print("\nDownloading Yahoo Finance data...")
    prices = download_prices(tickers, LOOKBACK_YEARS)

    asset_prices = prices[ASSETS].copy()
    benchmark_prices = prices[BENCHMARK].copy()

    asset_returns = calculate_returns(asset_prices)
    benchmark_returns = calculate_returns(benchmark_prices)

    # Align benchmark to asset-return dates.
    benchmark_returns = benchmark_returns.reindex(asset_returns.index).dropna()
    asset_returns = asset_returns.reindex(benchmark_returns.index).dropna()
    asset_prices = asset_prices.reindex(asset_returns.index).dropna()

    print(f"Observations: {len(asset_returns):,}")
    print(f"Start date:   {asset_returns.index.min().date()}")
    print(f"End date:     {asset_returns.index.max().date()}")

    # Covariance
    cov, shrinkage = ledoit_wolf_covariance(asset_returns)
    historical_returns = annualized_mean(asset_returns)

    # Market weights
    mkt_w = market_weights(ASSETS, asset_returns, asset_prices)

    # Risk-free rate
    RISK_FREE_RATE = get_risk_free_rate()

    # Market expected return for delta.
    market_return = float(mkt_w.values @ historical_returns.values)

    if RISK_AVERSION_METHOD == "fixed":
        delta = FIXED_RISK_AVERSION
    else:
        delta = market_implied_risk_aversion(mkt_w, cov, market_return)

    # Equilibrium returns
    pi = pd.Series(
        delta * (cov.values @ mkt_w.values),
        index=ASSETS
    )

    # Views
    P, Q = build_views(VIEWS, ASSETS)

    if REQUIRE_VIEWS and (P is None or Q is None):
        raise RuntimeError("Views are required but VIEWS is empty.")

    # Tau and posterior
    if P is None:
        tau = np.nan
        omega = None
        posterior = pi.copy()
    else:
        tau, _ = estimate_tau_bootstrap(
            asset_returns,
            mkt_w,
            delta,
            cov
        )

        if OMEGA_METHOD != "he_litterman":
            raise ValueError("OMEGA_METHOD must be 'he_litterman'.")

        omega = tau * P @ cov.values @ P.T

        # Numerical stabilization for one or more views.
        omega = np.atleast_2d(omega)
        omega += np.eye(omega.shape[0]) * 1e-12

        posterior = black_litterman_posterior(
            pi,
            cov,
            tau,
            P,
            Q,
            omega
        )

    # Optimize
    weights = optimize_portfolio(posterior, cov)

    # Efficient frontier
    frontier_vol, frontier_ret = efficient_frontier(
        posterior,
        cov,
        N_EFFICIENT_PORTFOLIOS
    )

    # Portfolio risk metrics
    portfolio_daily = portfolio_daily_returns(weights, asset_returns)
    portfolio_metrics = annualized_metrics(portfolio_daily)
    portfolio_dd, portfolio_drawdown = max_drawdown(portfolio_daily)
    var_95, cvar_95 = historical_var_cvar(
        portfolio_daily,
        VAR_CONFIDENCE
    )

    # Benchmark metrics
    benchmark_metrics = annualized_metrics(benchmark_returns)
    benchmark_dd, benchmark_drawdown = max_drawdown(benchmark_returns)

    # ========================================================
    # REPORT
    # ========================================================

    print("\nCONFIGURATION")
    print("-" * 13)
    print(f"Assets:                 {', '.join(ASSETS)}")
    print(f"Benchmark:              {BENCHMARK}")
    print(f"Lookback:               {LOOKBACK_YEARS} year(s)")
    print(f"Frequency:              Daily")
    print(f"Market portfolio:       {MARKET_PORTFOLIO_METHOD}")
    print(f"Risk-free method:       {RISK_FREE_METHOD}")
    print(f"Risk-free rate:         {RISK_FREE_RATE:.4%}")
    print(f"Risk aversion method:   {RISK_AVERSION_METHOD}")
    print(f"Risk aversion (delta):  {delta:.6f}")
    print(f"Covariance:             Ledoit-Wolf")
    print(f"Shrinkage coefficient:  {shrinkage:.6f}")
    print(f"Optimization:           {OPTIMIZATION_OBJECTIVE}")
    print(f"Short selling:          {ALLOW_SHORT}")
    print(f"Weight range:           [{MIN_WEIGHT:.2%}, {MAX_WEIGHT:.2%}]")
    print(f"VaR/CVaR confidence:    {VAR_CONFIDENCE:.0%}")

    print_vector("MARKET WEIGHTS", mkt_w)
    print_vector("HISTORICAL ANNUALIZED MEAN RETURNS", historical_returns)
    print_matrix("ANNUALIZED LEDOIT-WOLF COVARIANCE MATRIX", cov)
    print_vector("EQUILIBRIUM RETURNS (PI)", pi)

    if P is not None:
        print("\nINVESTOR VIEWS")
        print("-" * 14)
        print("P matrix:")
        print(pd.DataFrame(P, columns=ASSETS).to_string(index=False))
        print("\nQ vector:")
        print(Q)
        print(f"\nTau: {tau:.8f}")
        print("\nOmega:")
        print(pd.DataFrame(omega).to_string())

    print_vector("BLACK-LITTERMAN POSTERIOR RETURNS", posterior)
    print_vector("OPTIMAL PORTFOLIO WEIGHTS", weights)

    print("\nPORTFOLIO METRICS")
    print("-" * 17)
    print(f"Expected annual return: {portfolio_metrics['annual_return']:.2%}")
    print(f"Annual volatility:      {portfolio_metrics['annual_volatility']:.2%}")
    print(f"Sharpe ratio:            {portfolio_metrics['sharpe']:.4f}")
    print(f"Cumulative return:       {portfolio_metrics['cumulative_return']:.2%}")
    print(f"Maximum drawdown:        {portfolio_dd:.2%}")
    print(f"Historical VaR (95%):    {var_95:.2%}")
    print(f"Historical CVaR (95%):   {cvar_95:.2%}")

    print("\nS&P 500 BENCHMARK METRICS")
    print("-" * 25)
    print(f"Annualized return:       {benchmark_metrics['annual_return']:.2%}")
    print(f"Annual volatility:       {benchmark_metrics['annual_volatility']:.2%}")
    print(f"Sharpe ratio:            {benchmark_metrics['sharpe']:.4f}")
    print(f"Cumulative return:       {benchmark_metrics['cumulative_return']:.2%}")
    print(f"Maximum drawdown:        {benchmark_dd:.2%}")

    print("\nDisplaying charts...")
    plot_results(
        asset_prices,
        asset_returns,
        weights,
        pi,
        posterior,
        frontier_vol,
        frontier_ret,
        benchmark_returns,
        BENCHMARK
    )


if __name__ == "__main__":
    main()
