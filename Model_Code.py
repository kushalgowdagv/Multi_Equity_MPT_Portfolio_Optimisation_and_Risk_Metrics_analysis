


import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

import scipy.optimize as sco

import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Enhanced Portfolio VaR and CVaR Calculator", layout="wide")

# API Key
API_KEY = '0uTB4phKEr4dHcB2zJMmVmKUcywpkxDQ'

# Caching data fetching for efficiency
@st.cache_data
def fetch_stock_data(stock_name, start_date, end_date, API_KEY):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{stock_name}?from={start_date}&to={end_date}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if 'historical' not in data or not data['historical']:
            return None, f"No data available for {stock_name} in the selected range."
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df['close'].values, None
    except Exception as e:
        return None, f"Error fetching stock data: {str(e)}"

# Sidebar inputs
st.sidebar.header("Portfolio Configuration")
portfolio_value = st.sidebar.number_input("Portfolio Value (in USD):", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
n = st.sidebar.slider("Number of Stocks (up to 10):", 1, 10, 2)

stock_names = []
weights = [100.0]

for i in range(n):
    if i >= len(weights):
        weights.append(0.0)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        stock_name = st.text_input(f"Stock Ticker {i + 1}:", key=f"stock_{i}")
    with col2:
        weight = st.number_input(f"Weight (%) {i + 1}:", value=weights[i], min_value=0.0, max_value=100.0, step=1.0, key=f"weight_{i}")
        weights[i] = weight

    stock_names.append(stock_name)

    # Adjust weights dynamically
    if sum(weights) > 100.0:
        diff = sum(weights) - 100.0
        weights[i] -= diff
        st.sidebar.warning("Weights have been automatically adjusted to sum to 100%.")

# Display the total weight
st.sidebar.write(f"Total Weight: {sum(weights):.2f}%")


# Set end date to today's date and start date to one year before
default_end_date = datetime.today()
default_start_date = default_end_date - timedelta(days=365)

# Sidebar inputs for date range
start_date = st.sidebar.date_input("Start Date:", default_start_date)
end_date = st.sidebar.date_input("End Date:", default_end_date)

# Validate the date inputs
if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")

# Fetch and process stock data
# if st.sidebar.button("Fetch Data"):
#     if abs(sum(weights) - 100) > 0.01:
#         st.sidebar.error("Weights must sum to 100%.")
#     else:
#         portfolio_returns = []
#         all_returns = {}
#         for i, stock_name in enumerate(stock_names):
#             stock_data, error_message = fetch_stock_data(stock_name, start_date.isoformat(), end_date.isoformat(), API_KEY)
#             if stock_data is not None:
#                 returns = np.diff(stock_data) / stock_data[:-1]
#                 weighted_returns = returns * weights[i] / 100
#                 portfolio_returns.append(weighted_returns)
#                 all_returns[stock_name] = returns
#             else:
#                 st.error(error_message)
#                 break

#         if portfolio_returns:
#             portfolio_returns = np.sum(portfolio_returns, axis=0)
#             st.session_state['portfolio_returns'] = portfolio_returns
#             st.session_state['all_returns'] = all_returns
#             st.success("Data fetched successfully!")
#         else:
#             st.error("Failed to fetch stock data.")

# Fetch and process stock data
if st.sidebar.button("Fetch Data"):
    if abs(sum(weights) - 100) > 0.01:
        st.sidebar.error("Weights must sum to 100%.")
    else:
        portfolio_returns = []
        all_returns = {}
        min_length = None  # To track the shortest length of stock returns
        for i, stock_name in enumerate(stock_names):
            stock_data, error_message = fetch_stock_data(stock_name, start_date.isoformat(), end_date.isoformat(), API_KEY)
            if stock_data is not None:
                returns = np.diff(stock_data) / stock_data[:-1]
                if min_length is None or len(returns) < min_length:
                    min_length = len(returns)  # Update the minimum length
                weighted_returns = returns * weights[i] / 100
                portfolio_returns.append(weighted_returns)
                all_returns[stock_name] = returns
            else:
                st.error(error_message)
                break

        # Truncate all returns to the same length
        if portfolio_returns:
            portfolio_returns = [r[:min_length] for r in portfolio_returns]
            portfolio_returns = np.sum(portfolio_returns, axis=0)
            st.session_state['portfolio_returns'] = portfolio_returns
            st.session_state['all_returns'] = {
                name: ret[:min_length] for name, ret in all_returns.items()
            }
            st.success("Data fetched successfully!")
        else:
            st.error("Failed to fetch stock data.")


# Helper function for metrics
@st.cache_data
def calculate_metrics(portfolio_returns):
    mean_return = np.mean(portfolio_returns)
    std_dev = np.std(portfolio_returns)
    sharpe_ratio = mean_return / std_dev if std_dev != 0 else 0
    return mean_return, std_dev, sharpe_ratio

# VaR and CVaR calculations
def calculate_var(returns, confidence_level, method, portfolio_value):
    if method == "Historical":
        var = np.percentile(returns, 100 - confidence_level)
    elif method == "Variance-Covariance":
        mean = np.mean(returns)
        sigma = np.std(returns)
        z_score = -(stats.norm.ppf(1 - confidence_level / 100))
        var = -(mean + z_score * sigma)
    elif method == "Monte Carlo":
        simulations = 10000
        mean = np.mean(returns)
        sigma = np.std(returns)
        simulated_returns = np.random.normal(mean, sigma, simulations)
        a = -(np.percentile(simulated_returns, 100 - confidence_level))
        var= -(a)
    return var * portfolio_value 

def calculate_cvar(returns, confidence_level, method, portfolio_value):
    var = calculate_var(returns, confidence_level, method, portfolio_value)
    cvar = np.mean(returns[returns <= var])
    return cvar

def plot_returns(returns):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=returns, mode='lines', name='Portfolio Returns'))
    fig.update_layout(title="Portfolio Returns Over Time",
                      xaxis_title="Days",
                      yaxis_title="Returns")
    st.plotly_chart(fig)

def plot_correlation_heatmap(all_returns):
    returns_df = pd.DataFrame(all_returns)
    correlation_matrix = returns_df.corr()
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                     x=correlation_matrix.columns,
                                     y=correlation_matrix.columns,
                                     colorscale='viridis',
                                     zmin=-1, zmax=1))
    fig.update_layout(title="Correlation Heatmap of Stock Returns")
    st.plotly_chart(fig)

def plot_cvar_distribution(returns, var, cvar):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns', opacity=0.7))
    fig.add_vline(x=var, line_width=2, line_dash="dash", line_color="red", name=f"VaR ({var:.2f})")
    fig.add_vline(x=cvar, line_width=2, line_dash="dash", line_color="orange", name=f"CVaR ({cvar:.2f})")
    fig.update_layout(title="VaR and CVaR Distribution",
                      xaxis_title="Returns",
                      yaxis_title="Frequency",
                      legend_title="Metrics")
    st.plotly_chart(fig)

# Risk metric calculations
# if st.button("Calculate Metrics"):
#     if 'portfolio_returns' in st.session_state:
#         returns = st.session_state['portfolio_returns']

#         # Basic metrics
#         mean_return, std_dev, sharpe_ratio = calculate_metrics(returns)
#         st.write(f"Mean Return: {mean_return:.4f}")
#         st.write(f"Standard Deviation: {std_dev:.4f}")
#         st.write(f"Sharpe Ratio: {sharpe_ratio:.4f}")

#         # VaR and CVaR
#         confidence_level = st.slider("Confidence Level:", 90, 99, 95)
#         var = calculate_var(returns, confidence_level)
#         cvar = calculate_cvar(returns, confidence_level)
#         st.write(f"VaR at {confidence_level}%: {var:.4f}")
#         st.write(f"CVaR at {confidence_level}%: {cvar:.4f}")

#         # Plot returns and CVaR
#         plot_returns(returns)
#         plot_cvar_distribution(returns, var, cvar)

#         # Correlation heatmap
#         plot_correlation_heatmap(st.session_state['all_returns'])
#     else:
#         st.error("Please fetch data first.")

# Expanded Risk Metrics
def calculate_beta(portfolio_returns, market_returns):
    """
    Calculate Beta to assess systematic risk.
    """
    covariance_matrix = np.cov(portfolio_returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]  # Cov(portfolio, market) / Var(market)
    return beta

def calculate_max_drawdown(portfolio_cumulative_returns):
    """
    Calculate the maximum drawdown of the portfolio.
    """
    running_max = np.maximum.accumulate(portfolio_cumulative_returns)
    drawdowns = (portfolio_cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    return max_drawdown

def calculate_sortino_ratio(portfolio_returns, risk_free_rate, target_return=0.0):
    """
    Calculate the Sortino Ratio focusing on downside risk-adjusted returns.
    """
    downside_deviation = np.std(portfolio_returns[portfolio_returns < target_return])
    mean_return = np.mean(portfolio_returns)
    sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
    return sortino_ratio

# Updated workflow to calculate Expanded Risk Metrics
# if st.button("Calculate Expanded Risk Metrics"):
#     if 'portfolio_returns' in st.session_state:
#         portfolio_returns = st.session_state['portfolio_returns']

#         # Cumulative returns for drawdown analysis
#         portfolio_cumulative_returns = np.cumsum(portfolio_returns)

#         # Assume market returns as a proxy (e.g., S&P 500 index)
#         market_name = st.text_input("Enter Market Ticker (e.g., SPY):", value="SPY")
#         market_data, error_message = fetch_stock_data(market_name, start_date.isoformat(), end_date.isoformat(), API_KEY)
#         if market_data is not None:
#             market_returns = np.diff(market_data) / market_data[:-1]

#             # Beta calculation
#             beta = calculate_beta(portfolio_returns, market_returns)
#             st.write(f"**Portfolio Beta (Systematic Risk):** {beta:.4f}")
#         else:
#             st.error(f"Market data could not be fetched: {error_message}")

#         # Max Drawdown calculation
#         max_drawdown = calculate_max_drawdown(portfolio_cumulative_returns)
#         st.write(f"**Maximum Drawdown:** {max_drawdown:.4f}")

#         # Sortino Ratio calculation
#         sortino_ratio = calculate_sortino_ratio(portfolio_returns, risk_free_rate=0.02)
#         st.write(f"**Sortino Ratio (Downside Risk-Adjusted Return):** {sortino_ratio:.4f}")

#         # Portfolio-level VaR and CVaR
#         confidence_level = st.slider("Confidence Level for Expanded Metrics:", 90, 99, 95)
#         portfolio_var = calculate_var(portfolio_returns, confidence_level)
#         portfolio_cvar = calculate_cvar(portfolio_returns, confidence_level)
#         st.write(f"**Portfolio VaR at {confidence_level}% Confidence Level:** {portfolio_var:.4f}")
#         st.write(f"**Portfolio CVaR at {confidence_level}% Confidence Level:** {portfolio_cvar:.4f}")
#     else:
#         st.error("Please fetch data first.")

# Combined Risk Metrics Calculation
if st.button("Calculate Metrics"):
    if 'portfolio_returns' in st.session_state:
        returns = st.session_state['portfolio_returns']

        # Basic metrics
        mean_return, std_dev, sharpe_ratio = calculate_metrics(returns)
        st.write(f"### Basic Metrics")
        st.write(f"Mean Return: {mean_return:.4f}")
        st.write(f"Standard Deviation: {std_dev:.4f}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.4f}")

        # VaR and CVaR
        var_methods = ["Historical", "Variance-Covariance", "Monte Carlo"]
        selected_method = st.selectbox("Select VaR Method:", var_methods)
        confidence_level = st.slider("Confidence Level for VaR/CVaR:", 90, 99, 95, key="confidence_level_metrics")
        if st.button("Calculate VaR"):
            if 'portfolio_returns' in st.session_state and st.session_state['portfolio_returns'].size:
        # Plotting the returns
            # plot_returns(st.session_state['portfolio_returns'])
                plot_returns(returns)

        # Calculating VaR
        var = calculate_var(st.session_state['portfolio_returns'], confidence_level, selected_method, portfolio_value)
        # var = calculate_var(returns, confidence_level)
        cvar = calculate_cvar(returns, confidence_level, selected_method, portfolio_value)
        st.write(f"### VaR and CVaR")
        st.write(f"VaR at {confidence_level}%: {var:.4f}")
        st.write(f"CVaR at {confidence_level}%: {cvar:.4f}")

        # Plot returns and CVaR
        
        plot_cvar_distribution(returns, var, cvar)

        # Correlation heatmap
        plot_correlation_heatmap(st.session_state['all_returns'])

        # Expanded Risk Metrics
        st.write(f"### Expanded Risk Metrics")

        # Cumulative returns for drawdown analysis
        portfolio_cumulative_returns = np.cumsum(returns)

        # Assume market returns as a proxy (e.g., S&P 500 index)
        market_name = st.text_input("Enter Market Ticker (e.g., SPY):", value="SPY", key="market_ticker_metrics")
        market_data, error_message = fetch_stock_data(market_name, start_date.isoformat(), end_date.isoformat(), API_KEY)
        if market_data is not None:
            market_returns = np.diff(market_data) / market_data[:-1]

            # Beta calculation
            beta = calculate_beta(returns, market_returns)
            st.write(f"**Portfolio Beta (Systematic Risk):** {beta:.4f}")
        else:
            st.error(f"Market data could not be fetched: {error_message}")

        # Max Drawdown calculation
        max_drawdown = calculate_max_drawdown(portfolio_cumulative_returns)
        st.write(f"**Maximum Drawdown:** {max_drawdown:.4f}")

        # Sortino Ratio calculation
        sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        st.write(f"**Sortino Ratio (Downside Risk-Adjusted Return):** {sortino_ratio:.4f}")
    else:
        st.error("Please fetch data first.")


# Modern Portfolio Theory Optimization
@st.cache_data
def calculate_optimized_weights(all_returns, risk_free_rate=0.02):
    """
    Perform portfolio optimization to maximize the Sharpe ratio.
    """
    # Convert returns to a DataFrame
    returns_df = pd.DataFrame(all_returns)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    num_assets = len(mean_returns)
    
    # Objective function: negative Sharpe ratio
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1

    # Initial guess (equal weighting)
    initial_weights = np.array(num_assets * [1.0 / num_assets])

    # Optimization
    optimized = sco.minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return optimized.x

# Sidebar for optimization options
st.sidebar.header("Portfolio Optimization")
optimize_portfolio = st.sidebar.checkbox("Optimize Portfolio (MPT)", value=False)

# Perform optimization if selected
# Visualization: Pie chart for initial and optimized weights
def plot_pie_charts(initial_weights, optimized_weights, stock_names):
    initial_weights_df = pd.DataFrame({
        'Stock': stock_names,
        'Weight (%)': initial_weights * 100
    })
    optimized_weights_df = pd.DataFrame({
        'Stock': stock_names,
        'Weight (%)': optimized_weights * 100
    })

    # Pie chart for initial weights
    fig_initial = px.pie(initial_weights_df, names='Stock', values='Weight (%)', title="Initial Portfolio Allocation")
    st.plotly_chart(fig_initial)

    # Pie chart for optimized weights
    fig_optimized = px.pie(optimized_weights_df, names='Stock', values='Weight (%)', title="Optimized Portfolio Allocation")
    st.plotly_chart(fig_optimized)

# Line chart for individual stock prices
def plot_stock_prices(stock_data, stock_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['close'],
        mode='lines',
        name=stock_name
    ))
    fig.update_layout(
        title=f"Closing Prices for {stock_name}",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    st.plotly_chart(fig)

# Portfolio performance comparison before and after optimization
def plot_portfolio_performance(portfolio_returns_before, portfolio_returns_after):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=np.cumsum(portfolio_returns_before),
        mode='lines',
        name="Portfolio (Before Optimization)"
    ))
    fig.add_trace(go.Scatter(
        y=np.cumsum(portfolio_returns_after),
        mode='lines',
        name="Portfolio (After Optimization)"
    ))
    fig.update_layout(
        title="Portfolio Performance Comparison (Cumulative Returns)",
        xaxis_title="Days",
        yaxis_title="Cumulative Returns"
    )
    st.plotly_chart(fig)

# # Main optimization workflow
# if st.sidebar.button("Run Optimization") and optimize_portfolio:
#     if 'all_returns' in st.session_state:
#         all_returns = st.session_state['all_returns']

#         # Calculate optimized weights
#         optimized_weights = calculate_optimized_weights(all_returns)

#         # Display pie charts for weights
#         st.write("### Portfolio Allocation Before and After Optimization")
#         initial_weights = np.array([weight / 100 for weight in weights[:len(all_returns)]])
#         plot_pie_charts(initial_weights, optimized_weights, list(all_returns.keys()))

#         # Plot stock price time series for each stock
#         st.write("### Stock Price Time Series")
#         for stock_name in all_returns.keys():
#             stock_data, _ = fetch_stock_data(stock_name, start_date.isoformat(), end_date.isoformat(), API_KEY)
#             if stock_data is not None:
#                 stock_df = pd.DataFrame({'close': stock_data})
#                 stock_df.index = pd.date_range(start=start_date, periods=len(stock_data))
#                 plot_stock_prices(stock_df, stock_name)

#         # Portfolio metrics and performance
#         returns_df = pd.DataFrame(all_returns)
#         mean_returns = returns_df.mean()
#         cov_matrix = returns_df.cov()

#         portfolio_return_before = np.dot(initial_weights, mean_returns)
#         portfolio_volatility_before = np.sqrt(np.dot(initial_weights.T, np.dot(cov_matrix, initial_weights)))
#         sharpe_ratio_before = (portfolio_return_before - 0.02) / portfolio_volatility_before  # Assuming risk-free rate = 0.02

#         portfolio_return_after = np.dot(optimized_weights, mean_returns)
#         portfolio_volatility_after = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
#         sharpe_ratio_after = (portfolio_return_after - 0.02) / portfolio_volatility_after

#         st.write("### Portfolio Metrics")
#         st.write(f"**Before Optimization:**")
#         st.write(f"Return: {portfolio_return_before:.4f}, Volatility: {portfolio_volatility_before:.4f}, Sharpe Ratio: {sharpe_ratio_before:.4f}")

#         st.write(f"**After Optimization:**")
#         st.write(f"Return: {portfolio_return_after:.4f}, Volatility: {portfolio_volatility_after:.4f}, Sharpe Ratio: {sharpe_ratio_after:.4f}")

#         # Portfolio performance comparison
#         st.write("### Portfolio Performance Comparison")
#         portfolio_returns_before = np.dot(returns_df.values, initial_weights)
#         portfolio_returns_after = np.dot(returns_df.values, optimized_weights)
#         plot_portfolio_performance(portfolio_returns_before, portfolio_returns_after)
#     else:
#         st.error("Please fetch data first.")


# Efficient Frontier Calculation
@st.cache_data
def calculate_efficient_frontier(all_returns, risk_free_rate=0.00):
    """
    Calculate the efficient frontier for the portfolio.
    """
    returns_df = pd.DataFrame(all_returns)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(mean_returns)

    # Generate random portfolios
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio

    return results, weights_record

# Plot Efficient Frontier
def plot_efficient_frontier(results, optimized_return, optimized_volatility):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results[1, :],
        y=results[0, :],
        mode='markers',
        marker=dict(color=results[2, :], colorscale='Viridis', showscale=True),
        name='Random Portfolios'
    ))
    fig.add_trace(go.Scatter(
        x=[optimized_volatility],
        y=[optimized_return],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Optimized Portfolio'
    ))
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility (Risk)",
        yaxis_title="Return",
        legend_title="Portfolios"
    )
    st.plotly_chart(fig)

# Multiple Optimization Strategies
def optimize_portfolio_strategy(all_returns, strategy='sharpe', risk_free_rate=0.02):
    """
    Optimize portfolio based on the selected strategy.
    """
    returns_df = pd.DataFrame(all_returns)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(mean_returns)

    # Objective functions
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def negative_return(weights):
        return -np.dot(weights, mean_returns)

    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess
    initial_weights = np.array(num_assets * [1.0 / num_assets])

    # Optimization
    if strategy == 'sharpe':
        optimized = sco.minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == 'min_volatility':
        optimized = sco.minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == 'max_return':
        optimized = sco.minimize(negative_return, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return optimized.x

# Sidebar for optimization strategy
st.sidebar.header("Optimization Strategy")
strategy = st.sidebar.selectbox(
    "Select Optimization Strategy:",
    options=['sharpe', 'min_volatility', 'max_return'],
    format_func=lambda x: 'Maximize Sharpe Ratio' if x == 'sharpe' else 'Minimize Volatility' if x == 'min_volatility' else 'Maximize Return'
)

# Main optimization workflow with new features
if st.sidebar.button("Run Optimization") and optimize_portfolio:
    if 'all_returns' in st.session_state:
        all_returns = st.session_state['all_returns']

        # Calculate efficient frontier
        results, weights_record = calculate_efficient_frontier(all_returns)

        # Optimize portfolio based on selected strategy
        optimized_weights = optimize_portfolio_strategy(all_returns, strategy=strategy)

        # Calculate metrics for optimized portfolio
        returns_df = pd.DataFrame(all_returns)
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        optimized_return = np.dot(optimized_weights, mean_returns)
        optimized_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))

        # Plot efficient frontier
        st.write("### Efficient Frontier")
        plot_efficient_frontier(results, optimized_return, optimized_volatility)

        # Display pie charts for weights
        st.write("### Portfolio Allocation Before and After Optimization")
        initial_weights = np.array([weight / 100 for weight in weights[:len(all_returns)]])
        plot_pie_charts(initial_weights, optimized_weights, list(all_returns.keys()))

        # Portfolio metrics and performance
        portfolio_return_before = np.dot(initial_weights, mean_returns)
        portfolio_volatility_before = np.sqrt(np.dot(initial_weights.T, np.dot(cov_matrix, initial_weights)))
        sharpe_ratio_before = (portfolio_return_before - 0.02) / portfolio_volatility_before

        portfolio_return_after = np.dot(optimized_weights, mean_returns)
        portfolio_volatility_after = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
        sharpe_ratio_after = (portfolio_return_after - 0.02) / portfolio_volatility_after

        # st.write("### Portfolio Metrics")
        # st.write(f"**Before Optimization:**")
        # st.write(f"Return: {portfolio_return_before:.4f}, Volatility: {portfolio_volatility_before:.4f}, Sharpe Ratio: {sharpe_ratio_before:.4f}")

        # st.write(f"**After Optimization:**")
        # st.write(f"Return: {portfolio_return_after:.4f}, Volatility: {portfolio_volatility_after:.4f}, Sharpe Ratio: {sharpe_ratio_after:.4f}")

        metrics_data = {
            "Metric": ["Return (%)", "Volatility (%)", "Sharpe Ratio"],
            "Before Optimization": [
                portfolio_return_before * 100,  # Convert to percentage
                portfolio_volatility_before * 100,  # Convert to percentage
                sharpe_ratio_before
            ],
            "After Optimization": [
                portfolio_return_after * 100,  # Convert to percentage
                portfolio_volatility_after * 100,  # Convert to percentage
                sharpe_ratio_after
            ]
            }

# Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_data)

# Display metrics in table format
        st.write("### Portfolio Metrics Comparison")
        st.table(metrics_df)

        # Portfolio performance comparison
        st.write("### Portfolio Performance Comparison")
        portfolio_returns_before = np.dot(returns_df.values, initial_weights)
        portfolio_returns_after = np.dot(returns_df.values, optimized_weights)
        plot_portfolio_performance(portfolio_returns_before, portfolio_returns_after)
    else:
        st.error("Please fetch data first.")
