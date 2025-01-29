
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


def normalize_weights(weights):
    """Normalize weights to ensure they are whole numbers summing to 100."""
    weights = np.round(weights * 100)
    diff = 100 - np.sum(weights)
    if diff != 0:
        max_index = np.argmax(weights) if diff > 0 else np.argmin(weights)
        weights[max_index] += diff
    return weights

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

    optimized_weights = optimized.x if optimized.success else initial_weights
    optimized_weights = normalize_weights(optimized_weights)
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

        # Save optimized weights to session state
        st.session_state['optimized_weights'] = optimized_weights

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




# def calculate_var_cvar(portfolio_returns, confidence_level=0.95):
#     """Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)"""
#     sorted_returns = np.sort(portfolio_returns)
#     index = int((1 - confidence_level) * len(sorted_returns))
#     var = abs(sorted_returns[index])
#     cvar = abs(sorted_returns[:index].mean())
#     return var, cvar


# st.sidebar.header("Risk Analytics")
# risk_analysis = st.sidebar.checkbox("Run Risk Analytics")
# confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95) / 100

# if st.sidebar.button("Run Risk Analytics"):
#     if 'all_returns' in st.session_state:
#         all_returns = st.session_state['all_returns']
#         returns_df = pd.DataFrame(all_returns).dropna()
        
#         # Check if optimization is enabled and optimized weights are available
#         if optimize_portfolio and 'optimized_weights' in st.session_state:
#             weights = st.session_state['optimized_weights']
#             st.write("Using optimized weights for risk analytics.")
#         else:
#             # Use initial weights provided by the user
#             weights = np.array([weight / 100 for weight in weights[:len(all_returns)]])
#             st.write("Using initial weights for risk analytics.")
        
#         # Calculate portfolio returns
#         portfolio_returns = np.dot(returns_df, weights)
        
#         # Calculate VaR and CVaR
#         var, cvar = calculate_var_cvar(portfolio_returns, confidence_level)

#         # Display risk metrics
#         st.write(f"### Portfolio Risk Metrics at {confidence_level * 100:.0f}% Confidence")
#         st.metric(label="Value at Risk (VaR)", value=f"{var:.4f}")
#         st.metric(label="Conditional Value at Risk (CVaR)", value=f"{cvar:.4f}")
        
#         # Plot portfolio return distribution
#         fig = px.histogram(portfolio_returns, nbins=50, title="Portfolio Return Distribution")
#         st.plotly_chart(fig)
#     else:
#         st.error("Please fetch data first.")


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

def calculate_var_cvar(portfolio_returns, confidence_level=0.95):
    """Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)"""
    sorted_returns = np.sort(portfolio_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = abs(sorted_returns[index])
    cvar = abs(sorted_returns[:index].mean())
    return var, cvar

def calculate_historical_var(portfolio_returns, confidence_level=0.95):
    """Calculate Historical VaR"""
    sorted_returns = np.sort(portfolio_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    return abs(sorted_returns[index])

def calculate_variance_covariance_var(portfolio_returns, confidence_level=0.95):
    """Calculate Variance-Covariance VaR"""
    mean_return = np.mean(portfolio_returns)
    std_dev = np.std(portfolio_returns)
    return abs(norm.ppf(1 - confidence_level, mean_return, std_dev))

def calculate_monte_carlo_var(portfolio_returns, confidence_level=0.95, num_simulations=10000):
    """Calculate Monte Carlo VaR"""
    mean_return = np.mean(portfolio_returns)
    std_dev = np.std(portfolio_returns)
    simulated_returns = np.random.normal(mean_return, std_dev, num_simulations)
    return abs(np.percentile(simulated_returns, 100 * (1 - confidence_level)))

def calculate_expected_shortfall(portfolio_returns, confidence_level=0.95):
    """Calculate Expected Shortfall (Conditional Value at Risk)"""
    sorted_returns = np.sort(portfolio_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    return abs(sorted_returns[:index].mean())

def calculate_maximum_drawdown(portfolio_returns):
    """Calculate Maximum Drawdown"""
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return np.min(drawdown)

def calculate_calmar_ratio(portfolio_returns, risk_free_rate=0.03):
    """Calculate Calmar Ratio"""
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    max_drawdown = calculate_maximum_drawdown(portfolio_returns)
    annualized_return = (cumulative_returns[-1] ** (252 / len(portfolio_returns))) - 1
    return (annualized_return - risk_free_rate) / abs(max_drawdown)

def calculate_rolling_volatility(portfolio_returns, window=30):
    """Calculate Rolling Volatility"""
    return portfolio_returns.rolling(window=window).std() * np.sqrt(252)

def calculate_beta(portfolio_returns, benchmark_returns):
    """Calculate Beta, ensuring both arrays have the same length."""
    # Ensure both arrays have the same length
    min_length = min(len(portfolio_returns), len(benchmark_returns))
    portfolio_returns = portfolio_returns[:min_length]
    benchmark_returns = benchmark_returns[:min_length]

    # Calculate covariance and variance
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns)
    
    # Avoid division by zero
    if variance == 0:
        return np.nan
    return covariance / variance

# def calculate_beta(portfolio_returns, benchmark_returns):
#     """Calculate Beta"""
#     covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
#     variance = np.var(benchmark_returns)
#     return covariance / variance

# def fetch_benchmark_data(benchmark_ticker, start_date, end_date, API_KEY):
#     """Fetch benchmark data from the API."""
#     # url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{benchmark_ticker}?from={start_date}&to={end_date}&apikey={API_KEY}"
#     url = f"https://financialmodelingprep.com/api/v3/quotes/index/{benchmark_ticker}?from={start_date}&to={end_date}&apikey={API_KEY}"

#     try:
#         response = requests.get(url)
#         data = response.json()
#         if 'historical' not in data or not data['historical']:
#             return None, f"No data available for {benchmark_ticker} in the selected range."
#         df = pd.DataFrame(data['historical'])
#         df['date'] = pd.to_datetime(df['date'])
#         df = df.sort_values('date')
#         df['returns'] = df['close'].pct_change()
#         return df[['date', 'returns']], None
#     except Exception as e:
#         return None, f"Error fetching benchmark data: {str(e)}"
def fetch_benchmark():
    """
    Fetch all available benchmarks with symbol, name, and currency (filtered to "USD").
    Returns a DataFrame with benchmark data.
    """
    url = f"https://financialmodelingprep.com/api/v3/symbol/available-indexes?apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if not data:
            return None, "No benchmark data available from the API."
        
        # Convert to DataFrame
        benchmarks_df = pd.DataFrame(data)
        
        # Filter benchmarks with currency "USD"
        benchmarks_df = benchmarks_df[benchmarks_df['currency'] == 'USD']
        
        # Select relevant columns
        benchmarks_df = benchmarks_df[['symbol', 'name', 'currency']]
        
        return benchmarks_df, None
    except Exception as e:
        return None, f"Error fetching benchmark data: {str(e)}"
    
import yfinance as yf

def fetch_benchmark_data(benchmark_ticker, start_date, end_date):
    """
    Fetch historical data for the selected benchmark symbol using yfinance.
    """
    try:
        # Download historical data using yfinance
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)
        
        # Check if data is empty
        if benchmark_data.empty:
            return None, f"No data available for {benchmark_ticker} in the selected range."
        
        # Calculate daily returns
        benchmark_data['returns'] = benchmark_data['Close'].pct_change()
        
        # Return the relevant columns
        return benchmark_data[['returns']], None
    except Exception as e:
        return None, f"Error fetching benchmark data: {str(e)}"

# def fetch_benchmark_data(benchmark_symbol, start_date, end_date , API_KEY=API_KEY):
#     """
#     Fetch historical data for the selected benchmark symbol.
#     """
#     url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{benchmark_symbol}?from={start_date}&to={end_date}&apikey={API_KEY}"
#     try:
#         response = requests.get(url)
#         data = response.json()
#         if 'historical' not in data or not data['historical']:
#             return None, f"No data available for {benchmark_symbol} in the selected range."
        
#         # Convert to DataFrame
#         df = pd.DataFrame(data['historical'])
#         df['date'] = pd.to_datetime(df['date'])
#         df = df.sort_values('date')
#         df['returns'] = df['close'].pct_change()
        
#         return df[['date', 'returns']], None
#     except Exception as e:
#         return None, f"Error fetching benchmark data: {str(e)}"
    

# st.sidebar.header("Risk Analytics")
# risk_analysis = st.sidebar.checkbox("Run Risk Analytics")
# confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95) / 100
# rolling_window = st.sidebar.number_input("Rolling Volatility Window (Days):", min_value=1, value=30)
# benchmark_ticker = st.sidebar.selectbox("Select Benchmark Index:", options=['^GSPC', '^IXIC', '^DJI'], format_func=lambda x: "S&P 500" if x == "^GSPC" else "NASDAQ" if x == "^IXIC" else "Dow Jones")

# if st.sidebar.button("Run Risk Analytics"):
#     if 'all_returns' in st.session_state:
#         all_returns = st.session_state['all_returns']
#         returns_df = pd.DataFrame(all_returns).dropna()
        
#         # Check if optimization is enabled and optimized weights are available
#         if optimize_portfolio and 'optimized_weights' in st.session_state:
#             weights = st.session_state['optimized_weights']
#             st.write("Using optimized weights for risk analytics.")
#         else:
#             # Use initial weights provided by the user
#             weights = np.array([weight / 100 for weight in weights[:len(all_returns)]])
#             st.write("Using initial weights for risk analytics.")
        
#         # Calculate portfolio returns
#         portfolio_returns = np.dot(returns_df, weights)
        
#         # Calculate VaR and CVaR
#         var, cvar = calculate_var_cvar(portfolio_returns, confidence_level)

#         # Display risk metrics
#         st.write(f"### Portfolio Risk Metrics at {confidence_level * 100:.0f}% Confidence")
#         st.metric(label="Value at Risk (VaR)", value=f"{var:.4f}")
#         st.metric(label="Conditional Value at Risk (CVaR)", value=f"{cvar:.4f}")
        
#         # Plot portfolio return distribution
#         fig = px.histogram(portfolio_returns, nbins=50, title="Portfolio Return Distribution")
#         st.plotly_chart(fig)

#         # VaR Calculation Methods
#         st.write("### VaR Calculation Methods")
#         historical_var = calculate_historical_var(portfolio_returns, confidence_level)
#         variance_covariance_var = calculate_variance_covariance_var(portfolio_returns, confidence_level)
#         monte_carlo_var = calculate_monte_carlo_var(portfolio_returns, confidence_level)

#         st.metric(label="Historical VaR", value=f"{historical_var:.4f}")
#         st.metric(label="Variance-Covariance VaR", value=f"{variance_covariance_var:.4f}")
#         st.metric(label="Monte Carlo VaR", value=f"{monte_carlo_var:.4f}")

#         # Expected Shortfall
#         expected_shortfall = calculate_expected_shortfall(portfolio_returns, confidence_level)
#         st.write("### Expected Shortfall (Conditional Drawdown at Risk)")
#         st.metric(label="Expected Shortfall", value=f"{expected_shortfall:.4f}")

#         # Maximum Drawdown
#         max_drawdown = calculate_maximum_drawdown(portfolio_returns)
#         st.write("### Maximum Drawdown")
#         st.metric(label="Maximum Drawdown", value=f"{max_drawdown:.4f}")

#         # Calmar Ratio
#         calmar_ratio = calculate_calmar_ratio(portfolio_returns)
#         st.write("### Calmar Ratio")
#         st.metric(label="Calmar Ratio", value=f"{calmar_ratio:.4f}")

#         # Rolling Volatility
#         rolling_volatility = calculate_rolling_volatility(pd.Series(portfolio_returns), rolling_window)
#         st.write("### Rolling Volatility")
#         fig = px.line(rolling_volatility, title="Rolling Volatility")
#         st.plotly_chart(fig)

#         # Beta Analysis
#         benchmark_data, error_message = fetch_benchmark_data(benchmark_ticker, start_date.isoformat(), end_date.isoformat(), API_KEY)
#         if benchmark_data is not None:
#             benchmark_returns = benchmark_data['returns'].dropna().values
#             beta = calculate_beta(portfolio_returns, benchmark_returns)
#             st.write("### Beta Analysis")
#             st.metric(label=f"Beta (vs {benchmark_ticker})", value=f"{beta:.4f}")
#         else:
#             st.error(error_message)
#     else:
#         st.error("Please fetch data first.")

st.sidebar.header("Risk Analytics")
risk_analysis = st.sidebar.checkbox("Run Risk Analytics")
confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95) / 100
rolling_window = st.sidebar.number_input("Rolling Volatility Window (Days):", min_value=1, value=30)

# Fetch available benchmarks
benchmarks_df, error_message = fetch_benchmark()
if benchmarks_df is None:
    st.sidebar.error(error_message)
else:
    # Create a selectbox for benchmark selection
    benchmark_options = benchmarks_df.apply(lambda row: f"{row['symbol']} - {row['name']}", axis=1).tolist()
    selected_benchmark = st.sidebar.selectbox("Select Benchmark Index:", options=benchmark_options)

    # Extract the symbol from the selected benchmark
    benchmark_ticker = selected_benchmark.split(" - ")[0]

if st.sidebar.button("Run Risk Analytics"):
    if 'all_returns' in st.session_state:
        all_returns = st.session_state['all_returns']
        returns_df = pd.DataFrame(all_returns).dropna()
        
        # Check if optimization is enabled and optimized weights are available
        if optimize_portfolio and 'optimized_weights' in st.session_state:
            weights = st.session_state['optimized_weights']
            st.write("Using optimized weights for risk analytics.")
        else:
            # Use initial weights provided by the user
            weights = np.array([weight / 100 for weight in weights[:len(all_returns)]])
            st.write("Using initial weights for risk analytics.")
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_df, weights)
        
        # Calculate VaR and CVaR
        var, cvar = calculate_var_cvar(portfolio_returns, confidence_level)

        # Display risk metrics
        st.write(f"### Portfolio Risk Metrics at {confidence_level * 100:.0f}% Confidence")
        st.metric(label="Value at Risk (VaR)", value=f"{var:.4f}")
        st.metric(label="Conditional Value at Risk (CVaR)", value=f"{cvar:.4f}")
        
        # Plot portfolio return distribution
        fig = px.histogram(portfolio_returns, nbins=50, title="Portfolio Return Distribution")
        st.plotly_chart(fig)

        # VaR Calculation Methods
        st.write("### VaR Calculation Methods")
        historical_var = calculate_historical_var(portfolio_returns, confidence_level)
        variance_covariance_var = calculate_variance_covariance_var(portfolio_returns, confidence_level)
        monte_carlo_var = calculate_monte_carlo_var(portfolio_returns, confidence_level)

        st.metric(label="Historical VaR", value=f"{historical_var:.4f}")
        st.metric(label="Variance-Covariance VaR", value=f"{variance_covariance_var:.4f}")
        st.metric(label="Monte Carlo VaR", value=f"{monte_carlo_var:.4f}")

        # Expected Shortfall
        expected_shortfall = calculate_expected_shortfall(portfolio_returns, confidence_level)
        st.write("### Expected Shortfall (Conditional Drawdown at Risk)")
        st.metric(label="Expected Shortfall", value=f"{expected_shortfall:.4f}")

        # Maximum Drawdown
        max_drawdown = calculate_maximum_drawdown(portfolio_returns)
        st.write("### Maximum Drawdown")
        st.metric(label="Maximum Drawdown", value=f"{max_drawdown:.4f}")

        # Calmar Ratio
        calmar_ratio = calculate_calmar_ratio(portfolio_returns)
        st.write("### Calmar Ratio")
        st.metric(label="Calmar Ratio", value=f"{calmar_ratio:.4f}")

        # Rolling Volatility
        rolling_volatility = calculate_rolling_volatility(pd.Series(portfolio_returns), rolling_window)
        st.write("### Rolling Volatility")
        fig = px.line(rolling_volatility, title="Rolling Volatility")
        st.plotly_chart(fig)

        # Beta Analysis
    #     benchmark_data, error_message = fetch_benchmark_data(benchmark_ticker, start_date.isoformat(), end_date.isoformat())
    #     if benchmark_data is not None:
    #         benchmark_returns = benchmark_data['returns'].dropna().values
    #         beta = calculate_beta(portfolio_returns, benchmark_returns)
    #         st.write("### Beta Analysis")
    #         st.metric(label=f"Beta (vs {selected_benchmark})", value=f"{beta:.4f}")
    #     else:
    #         st.error(error_message)
    # else:
    #     st.error("Please fetch data first.")
# Beta Analysis
        benchmark_data, error_message = fetch_benchmark_data(benchmark_ticker, start_date.isoformat(), end_date.isoformat())
        if benchmark_data is not None:
            benchmark_returns = benchmark_data['returns'].dropna().values
            portfolio_returns_aligned = portfolio_returns[:len(benchmark_returns)]  # Align lengths
            beta = calculate_beta(portfolio_returns_aligned, benchmark_returns)
            st.write("### Beta Analysis")
            st.metric(label=f"Beta (vs {selected_benchmark})", value=f"{beta:.4f}")
        else:
            st.error(error_message)