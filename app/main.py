import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

# --- Quantitative Explanations ---
QUANT_EXPLANATIONS = {
    "Expected Return": "Weighted average of asset returns in a portfolio.",
    "Volatility": "Standard deviation of portfolio returns, accounting for asset correlations.",
    "Sharpe Ratio": "(Portfolio Return - Risk-Free Rate) / Portfolio Volatility. Measures risk-adjusted return.",
    "Efficient Frontier": "Set of portfolios offering the highest expected return for each level of risk.",
    "Minimum Variance Portfolio": "Portfolio with the lowest possible volatility.",
    "Maximum Sharpe Portfolio": "Portfolio with the highest Sharpe ratio (optimal risk-adjusted return)."
}

# --- Simulate Asset Returns ---
def simulate_asset_returns(n_assets=4, n_years=5, seed=42):
    np.random.seed(seed)
    means = np.random.uniform(0.06, 0.18, n_assets)
    stds = np.random.uniform(0.10, 0.25, n_assets)
    corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr[i, j] = corr[j, i] = np.random.uniform(0.2, 0.8)
    cov = np.outer(stds, stds) * corr
    returns = np.random.multivariate_normal(means, cov, n_years * 252)
    return pd.DataFrame(returns, columns=[f"Asset {i+1}" for i in range(n_assets)])

# --- Portfolio Statistics ---
def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe

# --- Generate Random Portfolios ---
def generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate, n_portfolios):
    n_assets = len(mean_returns)
    results = np.zeros((n_portfolios, 3 + n_assets))
    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        results[i, 0] = ret
        results[i, 1] = vol
        results[i, 2] = sharpe
        results[i, 3:] = weights
    columns = ['Return', 'Volatility', 'Sharpe'] + [f"w_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(results, columns=columns)

# --- Optimization Functions ---
def min_variance(mean_returns, cov_matrix):
    n = len(mean_returns)
    def port_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(port_vol, n*[1./n], bounds=bounds, constraints=constraints)
    return result.x

def max_sharpe(mean_returns, cov_matrix, risk_free_rate):
    n = len(mean_returns)
    def neg_sharpe(weights):
        ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        return -sharpe
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(neg_sharpe, n*[1./n], bounds=bounds, constraints=constraints)
    return result.x

# --- Streamlit App ---
st.set_page_config(page_title="Efficient Frontier & Sharpe Landscape Lab", layout="wide")
st.title("Efficient Frontier & Sharpe Landscape Lab")

with st.sidebar:
    st.header("Simulation Settings")
    n_assets = st.slider("Number of Assets", 2, 8, 4)
    n_portfolios = st.slider("Number of Portfolios", 500, 10000, 3000, step=100)
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, step=0.1) / 100
    st.markdown("---")
    st.subheader("Quantitative Explanations")
    for k, v in QUANT_EXPLANATIONS.items():
        st.markdown(f"**{k}:** {v}")

returns_df = simulate_asset_returns(n_assets=n_assets)
mean_returns = returns_df.mean()
cov_matrix = returns_df.cov()

portfolios = generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate, n_portfolios)

# Find min variance and max Sharpe portfolios
w_min_var = min_variance(mean_returns, cov_matrix)
w_max_sharpe = max_sharpe(mean_returns, cov_matrix, risk_free_rate)
ret_min_var, vol_min_var, sharpe_min_var = portfolio_stats(w_min_var, mean_returns, cov_matrix, risk_free_rate)
ret_max_sharpe, vol_max_sharpe, sharpe_max_sharpe = portfolio_stats(w_max_sharpe, mean_returns, cov_matrix, risk_free_rate)

# --- Plot Efficient Frontier ---
fig = px.scatter(
    portfolios, x='Volatility', y='Return', color='Sharpe',
    color_continuous_scale='Viridis',
    title="Efficient Frontier & Sharpe Ratio Landscape",
    labels={"Volatility": "Portfolio Volatility (Std)", "Return": "Portfolio Expected Return", "Sharpe": "Sharpe Ratio"},
    hover_data={f"w_{i+1}": True for i in range(n_assets)}
)
fig.add_scatter(x=[vol_min_var], y=[ret_min_var], mode='markers',
                marker=dict(color='red', size=14, symbol='star'),
                name='Minimum Variance')
fig.add_scatter(x=[vol_max_sharpe], y=[ret_max_sharpe], mode='markers',
                marker=dict(color='gold', size=14, symbol='star'),
                name='Maximum Sharpe')

st.plotly_chart(fig, use_container_width=True)

# --- Show Portfolio Details ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Minimum Variance Portfolio")
    st.write(f"Expected Return: {ret_min_var:.2%}")
    st.write(f"Volatility: {vol_min_var:.2%}")
    st.write(f"Sharpe Ratio: {sharpe_min_var:.2f}")
    st.write(pd.DataFrame({"Weight": w_min_var}, index=mean_returns.index).T)
with col2:
    st.subheader("Maximum Sharpe Portfolio")
    st.write(f"Expected Return: {ret_max_sharpe:.2%}")
    st.write(f"Volatility: {vol_max_sharpe:.2%}")
    st.write(f"Sharpe Ratio: {sharpe_max_sharpe:.2f}")
    st.write(pd.DataFrame({"Weight": w_max_sharpe}, index=mean_returns.index).T)

st.markdown("---")
st.subheader("Simulated Asset Returns (first 10 rows)")
st.dataframe(returns_df.head(10))
