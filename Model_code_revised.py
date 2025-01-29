

import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import scipy.optimize as sco
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Enhanced Portfolio VaR and CVaR Calculator", layout="wide")

API_KEY = '0uTB4phKEr4dHcB2zJMmVmKUcywpkxDQ'
risk_free_rate = 0.03  # 3% annual risk-free rate
daily_risk_free_rate = risk_free_rate / 252
trading_days = 252

@st.cache_data
def fetch_stock_data(stock_name, start_date, end_date, API_KEY):
    """Fetch historical stock data from the API."""
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{stock_name}?from={start_date}&to={end_date}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if 'historical' not in data or not data['historical']:
            return None, f"No data available for {stock_name} in the selected range."
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df[['date', 'close']], None
    except Exception as e:
        return None, f"Error fetching stock data: {str(e)}"

def calculate_cagr(start_value, end_value, periods):
    """Calculate the Compound Annual Growth Rate (CAGR)."""
    return (end_value / start_value) ** (1 / periods) - 1 if periods > 0 else np.nan

def fetch_portfolio_data(stock_names, weights, start_date, end_date):
    """Fetch stock data and calculate portfolio returns."""
    stock_data_dict = {}
    all_returns = {}
    min_length = None

    for i, stock_name in enumerate(stock_names):
        stock_data, error_message = fetch_stock_data(stock_name, start_date.isoformat(), end_date.isoformat(), API_KEY)
        if stock_data is not None:
            stock_data_dict[stock_name] = stock_data
            stock_data['returns'] = stock_data['close'].pct_change()
            returns = stock_data['returns'].dropna().values
            if min_length is None or len(returns) < min_length:
                min_length = len(returns)
            all_returns[stock_name] = returns
        else:
            return None, error_message

    return stock_data_dict, all_returns, min_length

def calculate_portfolio_metrics(stock_data_dict, all_returns, min_length, stock_names, weights, start_date, end_date):
    """Calculate portfolio metrics including CAGR, volatility, and Sharpe Ratio."""
    portfolio_returns = np.sum([all_returns[name][:min_length] * (weights[i] / 100) for i, name in enumerate(stock_names)], axis=0)
    portfolio_cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    df_returns = pd.DataFrame(all_returns).dropna()
    
    years = (end_date - start_date).days / 252
    cagr_data = [calculate_cagr(stock_data_dict[name]['close'].iloc[0], stock_data_dict[name]['close'].iloc[-1], years) for name in stock_names]
    portfolio_start = sum(stock_data_dict[name]['close'].iloc[0] * (weights[i] / 100) for i, name in enumerate(stock_names))
    portfolio_end = sum(stock_data_dict[name]['close'].iloc[-1] * (weights[i] / 100) for i, name in enumerate(stock_names))
    portfolio_cagr = calculate_cagr(portfolio_start, portfolio_end, years)

    sharpe_ratios = [
        ((df_returns[name].mean() - daily_risk_free_rate) / df_returns[name].std()) * np.sqrt(trading_days)
        if df_returns[name].std() != 0 else np.nan
        for name in stock_names
    ]

    portfolio_sharpe_ratio = ((portfolio_returns.mean() - daily_risk_free_rate) / portfolio_returns.std()) * np.sqrt(trading_days) if portfolio_returns.std() != 0 else np.nan

    metrics_data = {
        'Stock': stock_names + ['Portfolio'],
        'CAGR': cagr_data + [portfolio_cagr],
        'Volatility': [df_returns[name].std() for name in stock_names] + [portfolio_returns.std()],
        'Sharpe Ratio': sharpe_ratios + [portfolio_sharpe_ratio]
    }
    return portfolio_cumulative_returns, df_returns, metrics_data

# Streamlit Sidebar Configuration
st.sidebar.header("Portfolio Configuration")
portfolio_value = st.sidebar.number_input("Portfolio Value (in USD):", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
n = st.sidebar.slider("Number of Stocks (up to 10):", 1, 10, 2)

stock_names = []
weights = []

for i in range(n):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        stock_name = st.text_input(f"Stock Ticker {i + 1}:", key=f"stock_{i}")
    with col2:
        weight = st.number_input(f"Weight (%) {i + 1}:", min_value=0.0, max_value=100.0, step=10.0, key=f"weight_{i}")
    stock_names.append(stock_name)
    weights.append(weight)

if sum(weights) > 100:
    st.sidebar.error("Total weight cannot exceed 100%. Adjust weights.")
elif sum(weights) < 100:
    st.sidebar.warning("Total weight is less than 100%. Consider adjusting.")

st.sidebar.write(f"Total Weight: {sum(weights):.2f}%")

default_end_date = datetime.today()
default_start_date = default_end_date - timedelta(days=365)

start_date = st.sidebar.date_input("Start Date:", default_start_date)
end_date = st.sidebar.date_input("End Date:", default_end_date)

if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")

# Fetch and Display Data
if st.sidebar.button("Fetch Data"):
    stock_data_dict, all_returns, min_length = fetch_portfolio_data(stock_names, weights, start_date, end_date)

    if stock_data_dict:
        portfolio_cumulative_returns, df_returns, metrics_data = calculate_portfolio_metrics(
            stock_data_dict, all_returns, min_length, stock_names, weights, start_date, end_date
        )

        # Save fetched data in session state for optimization use
        st.session_state['all_returns'] = all_returns
        st.session_state['df_returns'] = df_returns
        st.session_state['metrics_data'] = metrics_data  # Store metrics data for later


        # Save fetched data in session state for optimization use
        st.session_state['all_returns'] = all_returns
        st.session_state['df_returns'] = df_returns
        st.session_state['metrics_data'] = metrics_data  # Store metrics data for later

        # Plot Stock Charts
        stock_charts = []
        for stock_name, stock_data in stock_data_dict.items():
            fig = px.line(stock_data, x='date', y='close', title=f'{stock_name} Price Movement')
            fig.update_layout(xaxis_title='Date', yaxis_title='Closing Price', showlegend=True)
            stock_charts.append(fig)

        for i in range(0, len(stock_charts), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(stock_charts):
                    with cols[j]:
                        st.plotly_chart(stock_charts[i + j])

        # Portfolio Cumulative Returns
        portfolio_fig = px.line(x=stock_data_dict[stock_names[0]]['date'][1:min_length+1], y=portfolio_cumulative_returns, title='Cumulative Portfolio Returns')
        portfolio_fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative Returns', showlegend=True)
        st.plotly_chart(portfolio_fig)

        # Correlation Heatmap
        correlation_matrix = df_returns.corr()
        heatmap_fig = px.imshow(correlation_matrix, text_auto=True, title="Stock Returns Correlation Heatmap", color_continuous_scale='RdBu_r')
        st.plotly_chart(heatmap_fig)

        # Metrics Table
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.format({"CAGR": "{:.4f}", "Volatility": "{:.4f}", "Sharpe Ratio": "{:.4f}"}))

        st.success("Data fetched successfully!")
    else:
        st.error("Failed to fetch stock data.")



@st.cache_data
def calculate_efficient_frontier(all_returns, risk_free_rate=0.00):
    """Calculate the efficient frontier for the portfolio."""
    returns_df = pd.DataFrame(all_returns)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(mean_returns)

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



def optimize_portfolio_strategy(all_returns, strategy='sharpe', risk_free_rate=0.02):
    """Optimize portfolio based on the selected strategy."""
    returns_df = pd.DataFrame(all_returns)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(mean_returns)

    def negative_sharpe(weights):
        return -((np.dot(weights, mean_returns) - risk_free_rate) / np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def negative_return(weights):
        return -np.dot(weights, mean_returns)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1.0 / num_assets])

    if strategy == 'sharpe':
        optimized = sco.minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == 'min_volatility':
        optimized = sco.minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == 'max_return':
        optimized = sco.minimize(negative_return, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Ensure optimized weights are valid and sum to 1
    if optimized.success:
        optimized_weights = optimized.x / np.sum(optimized.x)  # Normalize weights
    else:
        optimized_weights = initial_weights  # Use initial equal weights if optimization fails

    return optimized_weights

def plot_portfolio_performance(portfolio_returns_before, portfolio_returns_after):
    """Plot cumulative portfolio returns before and after optimization."""
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

# Sidebar configuration for optimization
st.sidebar.header("Portfolio Optimization")
optimize_portfolio = st.sidebar.checkbox("Optimize Portfolio")
strategy = st.sidebar.selectbox(
    "Optimization Strategy:",
    options=['sharpe', 'min_volatility', 'max_return'],
    format_func=lambda x: "Max Sharpe Ratio" if x == "sharpe" else "Min Volatility" if x == "min_volatility" else "Max Return"
)

if st.sidebar.button("Run Optimization") and optimize_portfolio:
    if 'all_returns' in st.session_state:
        all_returns = st.session_state['all_returns']
        results, _ = calculate_efficient_frontier(all_returns)

        optimized_weights = optimize_portfolio_strategy(all_returns, strategy=strategy)

        # 🔹 Ensure optimized weights sum exactly to 1
        optimized_weights /= np.sum(optimized_weights)

        # 🔹 Remove small negative values
        optimized_weights = np.maximum(optimized_weights, 0)

        # Convert returns data
        returns_df = pd.DataFrame(all_returns)
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        optimized_return = np.dot(optimized_weights, mean_returns)
        optimized_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))

        plot_efficient_frontier(results, optimized_return, optimized_volatility)

        st.write("### Portfolio Allocation Before and After Optimization")
        initial_weights = np.array([weight / 100 for weight in weights[:len(all_returns)]])

        # Create DataFrames for Pie Charts
        df_initial = pd.DataFrame({'Stock': list(all_returns.keys()), 'Weight': initial_weights * 100})

        # 🔹 Filter out stocks with zero weight in optimized allocation
        optimized_df = pd.DataFrame({'Stock': list(all_returns.keys()), 'Weight': optimized_weights * 100})
        optimized_df = optimized_df[optimized_df["Weight"] > 0]

        # Side-by-side Pie Charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(df_initial, names="Stock", values="Weight", title="Initial Portfolio Weights"))
        with col2:
            if not optimized_df.empty:
                st.plotly_chart(px.pie(optimized_df, names="Stock", values="Weight", title="Optimized Portfolio Weights"))
            else:
                st.warning("No stocks allocated in optimized portfolio. Adjust strategy.")

        # Portfolio Returns Comparison
        portfolio_return_before = np.dot(initial_weights, mean_returns)
        portfolio_volatility_before = np.sqrt(np.dot(initial_weights.T, np.dot(cov_matrix, initial_weights)))

        portfolio_returns_before = np.dot(returns_df, initial_weights)
        portfolio_returns_after = np.dot(returns_df, optimized_weights)

        plot_portfolio_performance(portfolio_returns_before, portfolio_returns_after)

        metrics_data = {
            "Metric": ["Return (%)", "Volatility (%)"],
            "Before Optimization": [portfolio_return_before * 100, portfolio_volatility_before * 100],
            "After Optimization": [optimized_return * 100, optimized_volatility * 100],
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.write("### Portfolio Metrics Comparison")
        st.table(metrics_df)

    else:
        st.error("Please fetch data first.")


def calculate_var_cvar(portfolio_returns, confidence_level=0.95):
    """Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)"""
    sorted_returns = np.sort(portfolio_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = abs(sorted_returns[index])
    cvar = abs(sorted_returns[:index].mean())
    return var, cvar

st.sidebar.header("Risk Analytics")
risk_analysis = st.sidebar.checkbox("Run Risk Analytics")
confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95) / 100

if st.sidebar.button("Run Risk Analytics"):
    if 'all_returns' in st.session_state:
        all_returns = st.session_state['all_returns']
        returns_df = pd.DataFrame(all_returns).dropna()
        
        if 'optimized_weights' in st.session_state:
            weights = st.session_state['optimized_weights']
        else:
            weights = np.array([weight / 100 for weight in weights[:len(all_returns)]])
        
        portfolio_returns = np.dot(returns_df, weights)
        var, cvar = calculate_var_cvar(portfolio_returns, confidence_level)

        st.write(f"### Portfolio Risk Metrics at {confidence_level * 100:.0f}% Confidence")
        st.metric(label="Value at Risk (VaR)", value=f"{var:.4f}")
        st.metric(label="Conditional Value at Risk (CVaR)", value=f"{cvar:.4f}")
        
        fig = px.histogram(portfolio_returns, nbins=50, title="Portfolio Return Distribution")
        st.plotly_chart(fig)
    else:
        st.error("Please fetch data first.")
