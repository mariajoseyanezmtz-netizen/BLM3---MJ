
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from main import run_model, ASSETS, BENCHMARK

st.set_page_config(page_title="Black-Litterman Portfolio", layout="wide")
st.title("Black-Litterman Portfolio Optimization")
st.caption("Yahoo Finance | AAPL, MSFT, NVDA, AMZN, SPY, QQQ | Benchmark: ^GSPC")

years = st.selectbox("Historical window", [1, 3, 5], index=2)
run = st.button("Run Black-Litterman model", type="primary")

if run:
    with st.spinner("Downloading Yahoo Finance data and running the model..."):
        result = run_model(years)

    if result["market_info"]["method"].startswith("equal-weight fallback"):
        st.warning(
            "Yahoo Finance did not expose usable market-cap metadata in this "
            "cloud run. The model used the explicitly configured equal-weight "
            "fallback instead of inventing market-cap values."
        )

    st.subheader("Optimal portfolio")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected return", f"{result['metrics']['annual_return']:.2%}")
    c2.metric("Volatility", f"{result['metrics']['annual_volatility']:.2%}")
    c3.metric("Sharpe", f"{result['metrics']['sharpe']:.3f}")
    c4.metric("Max drawdown", f"{result['metrics']['max_drawdown']:.2%}")

    st.dataframe(
        result["weights"].rename("Weight").to_frame().style.format("{:.2%}"),
        use_container_width=True,
    )

    st.subheader("Black-Litterman posterior returns")
    st.dataframe(
        result["posterior"].rename("Posterior Return").to_frame().style.format("{:.2%}"),
        use_container_width=True,
    )

    st.subheader("Market portfolio")
    st.write(result["market_info"]["method"])
    st.dataframe(
        result["market_weights"].rename("Market Weight").to_frame().style.format("{:.2%}"),
        use_container_width=True,
    )

    st.subheader("Risk")
    risk = {
        "Historical VaR 95%": result["metrics"]["var_95"],
        "Historical CVaR 95%": result["metrics"]["cvar_95"],
    }
    st.dataframe(
        __import__("pandas").Series(risk, name="Value").to_frame().style.format("{:.2%}"),
        use_container_width=True,
    )

    # Efficient frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(result["frontier_vol"], result["frontier_ret"], s=8)
    selected_vol = (result["weights"].values @ result["covariance"].values @ result["weights"].values) ** 0.5
    selected_ret = result["weights"].values @ result["posterior"].values
    ax.scatter(selected_vol, selected_ret, marker="*", s=180, label="Selected portfolio")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Expected annual return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # Correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(result["returns"].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Return Correlation Matrix")
    st.pyplot(fig, clear_figure=True)

    # Cumulative
    common = __import__("pandas").concat(
        [
            result["portfolio_daily"].rename("Black-Litterman"),
            result["benchmark_returns"].rename(BENCHMARK),
        ],
        axis=1,
    ).dropna()
    cumulative = (1 + common).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative.index, cumulative["Black-Litterman"], label="Black-Litterman")
    ax.plot(cumulative.index, cumulative[BENCHMARK], label=BENCHMARK)
    ax.set_title("Cumulative Performance vs S&P 500")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Model inputs")
    st.write(f"Ledoit-Wolf shrinkage: `{result['shrinkage']:.6f}`")
    st.write(f"Risk-free rate: `{result['risk_free_rate']:.2%}`")
    st.write(f"Risk aversion (delta): `{result['delta']:.6f}`")
    if result["tau"] == result["tau"]:
        st.write(f"Tau: `{result['tau']:.8f}`")
        st.dataframe(result["omega"])
else:
    st.info("Select 1, 3, or 5 years and click **Run Black-Litterman model**.")
