
# Efficient Frontier & Sharpe Landscape Lab

## Creator/Dev: tubakhxn

## What is this project about?
This project is a Streamlit application for portfolio optimization research and education. It simulates asset returns, generates random portfolios, and visualizes the efficient frontier and Sharpe ratio landscape. Users can interactively explore risk-return tradeoffs, minimum variance, and maximum Sharpe portfolios.

## Features
- Simulate multiple asset returns
- Generate random portfolios
- Compute expected return, volatility, and Sharpe ratio
- Plot the efficient frontier (volatility vs. return, colored by Sharpe ratio)
- Highlight minimum variance and maximum Sharpe portfolios
- Interactive UI sliders for risk-free rate and number of portfolios

## Tech Stack
- Python
- NumPy, Pandas
- Plotly
- Streamlit
- SciPy


## How to Fork
1. Click the **Fork** button at the top right of the GitHub repository page.
2. Clone your forked repository:
   ```bash
   git clone https://github.com/your-username/Efficient-Frontier-Sharpe-Landscape-Lab.git
   ```
3. Navigate to the project directory and follow the usage instructions below.

## Usage
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app/main.py
   ```

## Quantitative Explanations
- **Expected Return:** Weighted average of asset returns in a portfolio.
- **Volatility:** Standard deviation of portfolio returns, accounting for asset correlations.
- **Sharpe Ratio:** (Portfolio Return - Risk-Free Rate) / Portfolio Volatility. Measures risk-adjusted return.
- **Efficient Frontier:** Set of portfolios offering the highest expected return for each level of risk.
- **Minimum Variance Portfolio:** Portfolio with the lowest possible volatility.
- **Maximum Sharpe Portfolio:** Portfolio with the highest Sharpe ratio (optimal risk-adjusted return).
