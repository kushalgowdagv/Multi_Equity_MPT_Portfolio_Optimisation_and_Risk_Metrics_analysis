
import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import scipy.optimize as sco
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import norm
import yfinance as yf
from scipy.stats import genpareto

st.set_page_config(page_title="Enhanced Portfolio VaR and CVaR Calculator", layout="wide")
API_KEY = '0uTB4phKEr4dHcB2zJMmVmKUcywpkxDQ'
# API_KEY = 'GzMRn53zNLe0FWXmElNdp2RakWorNyVi'
risk_free_rate = 0.03  # 3% annual risk-free rate
daily_risk_free_rate = risk_free_rate / 252
trading_days = 252



st.sidebar.markdown(
    """
    <div style="margin-bottom: 50px; padding-top: 20px; border-top: 1px solid #ccc; font-style: italic; text-align: center;">
        <p>"Risk is the price of opportunity; manage it wisely, and the rewards will follow."</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add this code at the top of your Streamlit app, after setting the page config.

# Custom CSS for the header
st.markdown(
    """
    <style>
    .project-title {
        font-size: 36px !important;
        font-weight: bold !important;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 10px;
    }
    .project-by {
        font-size: 20px !important;
        font-style: italic;
        text-align: center;
        color: #5D6D7E;
        margin-top: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with the project title and your name
st.markdown(
    """
    <div class="project-title">Multi-Dimensional Portfolio Risk and Performance Analysis</div>
    <div class="project-by">Project by Kushal Gowda G V</div>
    """,
    unsafe_allow_html=True
)

if 'data_fetched' not in st.session_state:
    st.session_state['data_fetched'] = False

# Display introduction only if data has not been fetched
if not st.session_state['data_fetched']:
    st.markdown(
        """
        ## Introduction: How to Use the Portfolio Risk and Performance Analysis Tool
        Welcome to the **Multi-Dimensional Portfolio Risk and Performance Analysis** tool! This application is designed to help you analyze and optimize your investment portfolio by providing detailed insights into risk metrics, performance, and optimization strategies. Below is a step-by-step guide on how to use the tool effectively.

        ---

        ### Step 1: **Portfolio Configuration**
        1. **Portfolio Value**: Enter the total value of your portfolio in USD. For example, you can start with `$100,000`.
        2. **Number of Stocks**: Use the slider to select the number of stocks in your portfolio (up to 10). For example, select `2` for a two-stock portfolio.
        3. **Stock Tickers and Weights**:
           - Enter the stock tickers (e.g., `AAPL` for Apple, `MSFT` for Microsoft).
           - Assign weights to each stock as a percentage of the total portfolio. For example:
             - `AAPL`: 60%
             - `MSFT`: 40%
           - Ensure the total weight adds up to 100%. If not, the tool will prompt you to adjust the weights.

        ---

        ### Step 2: **Date Range**
        - Select a start and end date for the analysis. For example:
          - Start Date: `2022-01-01`
          - End Date: `2023-01-01`
        - Ensure the start date is earlier than the end date.

        ---

        ### Step 3: **Fetch Data**
        - Click the **"Fetch Data"** button to retrieve historical stock data for the selected tickers and date range.
        - Once the data is fetched, the tool will display:
          - Price movement charts for each stock.
          - Cumulative portfolio returns.
          - A correlation heatmap of stock returns.
          - Key metrics such as CAGR, Volatility, and Sharpe Ratio for each stock and the portfolio.

        ---

        ### Step 4: **Portfolio Optimization (Optional)**
        - If you want to optimize your portfolio, check the **"Optimize Portfolio"** checkbox.
        - Select an optimization strategy:
          - **Max Sharpe Ratio**: Maximizes the risk-adjusted return.
          - **Min Volatility**: Minimizes portfolio risk.
          - **Max Return**: Maximizes portfolio return.
        - Click **"Run Optimization"** to calculate the optimized weights.
        - The tool will display:
          - The efficient frontier.
          - A comparison of portfolio performance before and after optimization.
          - Updated portfolio weights.

        ---

        ### Step 5: **Risk Analytics**
        - Check the **"Run Risk Analytics"** checkbox to analyze portfolio risk.
        - Select a confidence level (e.g., `95%`) and a rolling volatility window (e.g., `30 days`).
        - Choose a benchmark index (e.g., `^GSPC` for S&P 500) for beta analysis.
        - Click **"Run Risk Analytics"** to calculate:
          - Value at Risk (VaR) and Conditional Value at Risk (CVaR).
          - Historical, Variance-Covariance, and Monte Carlo VaR.
          - Calmar Ratio, Maximum Drawdown, and Expected Shortfall.
          - Rolling Volatility, Drawdown Analysis, and Tail Risk.
          - Stress Testing and Monte Carlo Simulation results.

        ---

        ### Example Walkthrough:
        1. **Portfolio Configuration**:
           - Portfolio Value: `$100,000`
           - Number of Stocks: `2`
           - Stock Tickers: `AAPL` (60%), `MSFT` (40%)
        2. **Date Range**:
           - Start Date: `2022-01-01`
           - End Date: `2023-01-01`
        3. **Fetch Data**:
           - Click **"Fetch Data"** to load historical data and view initial metrics.
        4. **Optimization**:
           - Check **"Optimize Portfolio"** and select **"Max Sharpe Ratio"**.
           - Click **"Run Optimization"** to view optimized weights and performance.
        5. **Risk Analytics**:
           - Check **"Run Risk Analytics"**, set Confidence Level to `95%`, and select Benchmark Index `^GSPC`.
           - Click **"Run Risk Analytics"** to view detailed risk metrics and simulations.

        ---

        #### Key Notes:
        - If **"Optimize Portfolio"** is checked, the risk metrics will use the optimized weights. Otherwise, the initial weights will be used.
        - The introduction will disappear once you click **"Fetch Data"**.

        ---

        Feel free to explore the tool and experiment with different portfolios, optimization strategies, and risk analytics. If you have any questions or feedback, connect with me using the links provided in the sidebar or footer! 🚀

        ---

        This introduction will vanish once you click the **"Fetch Data"** button. Happy analyzing! 📊📈
        """,
        unsafe_allow_html=True
    )





def fetch_stock_data(stock_name, start_date, end_date, API_KEY):
    """
    Fetch historical stock data from Financial Modeling Prep. 
    If no data is returned, fallback to yfinance.
    """
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{stock_name}?from={start_date}&to={end_date}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check if FMP returned any valid data
        if 'historical' not in data or not data['historical']:
            # Fallback to yfinance
            fallback_data = yf.download(stock_name, start=start_date, end=end_date)
            if fallback_data.empty:
                return None, f"No data available from FMP or yfinance for {stock_name}."
            
            # Prepare dataframe to match the format used elsewhere
            fallback_data.reset_index(inplace=True)
            fallback_data.rename(columns={'Date': 'date', 'Close': 'close'}, inplace=True)
            fallback_data = fallback_data[['date', 'close']]
            return fallback_data, None
        
        # If FMP returns data
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df[['date', 'close']], None
    
    except Exception as e:
        # Also fallback to yfinance if there's an exception calling FMP
        fallback_data = yf.download(stock_name, start=start_date, end=end_date)
        if fallback_data.empty:
            return None, f"Error fetching stock data from FMP and no data on yfinance: {str(e)}"
        
        fallback_data.reset_index(inplace=True)
        fallback_data.rename(columns={'Date': 'date', 'Close': 'close'}, inplace=True)
        fallback_data = fallback_data[['date', 'close']]
        return fallback_data, None


# def calculate_cagr(start_value, end_value, periods):
#     """Calculate the Compound Annual Growth Rate (CAGR)."""
#     return (end_value / start_value) ** (1 / periods) - 1 if periods > 0 else np.nan

def calculate_cagr(start_value, end_value, periods):
    """Calculate the Compound Annual Growth Rate (CAGR)."""
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return None  # Avoid division errors and ensure valid values
    return (end_value / start_value) ** (1 / periods) - 1


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
            return None, None ,error_message
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



if st.sidebar.button("Fetch Data"):
    
    stock_data_dict, all_returns, min_length = fetch_portfolio_data(stock_names, weights, start_date, end_date)
    st.session_state['data_fetched'] = True

    if stock_data_dict:
        
        # Store stock_data_dict in session state
        st.session_state['stock_data_dict'] = stock_data_dict

        portfolio_cumulative_returns, df_returns, metrics_data = calculate_portfolio_metrics(
            stock_data_dict, all_returns, min_length, stock_names, weights, start_date, end_date
        )

        # Save fetched data in session state for optimization use
        st.session_state['all_returns'] = all_returns
        st.session_state['df_returns'] = df_returns
        st.session_state['metrics_data'] = metrics_data  # Store metrics data for later

        # Plot Stock Charts
        stock_charts = []
        for stock_name, stock_data in stock_data_dict.items():
            stock_data.columns = stock_data.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)

# Rename columns explicitly
            stock_data = stock_data.rename(columns={'date': 'Date', 'close': 'Close', 'returns': 'Returns'})

# Plot Stock Charts
            fig = px.line(stock_data, x='Date', y='Close', title=f'{stock_name} Price Movement')
            # fig = px.line(stock_data, x='date', y='close', title=f'{stock_name} Price Movement')
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
        # metrics_df["CAGR"] = pd.to_numeric(metrics_df["CAGR"], errors='coerce')
        metrics_df["CAGR"] = pd.to_numeric(metrics_df["CAGR"], errors='coerce')

        metrics_df["Volatility"] = pd.to_numeric(metrics_df["Volatility"], errors='coerce')
        metrics_df["Sharpe Ratio"] = pd.to_numeric(metrics_df["Sharpe Ratio"], errors='coerce')

        
        st.dataframe(metrics_df.style.format({"CAGR": "{:.4f}", "Volatility": "{:.4f}", "Sharpe Ratio": "{:.4f}"}))

        # metrics_df = metrics_df.apply(pd.to_numeric, errors='coerce')


        # st.dataframe(metrics_df.style.applymap(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"))
        st.success("Data fetched successfully!")
    else:
        st.error("Failed to fetch stock data.")


# @st.cache_data
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




def fetch_benchmark():
    """
    Return a list of benchmarks in 'Name - Ticker' format.
    """
    benchmark_dict = {
        "S&P 500": "^GSPC",
        "Dow Jones Industrial Average (DJIA)": "^DJI",
        "Nasdaq Composite": "^IXIC",
        "Russell 2000": "^RUT",
        "S&P 100": "^OEX",
        "Nasdaq 100": "^NDX",
        "NYSE Composite": "^NYA",
        "S&P MidCap 400": "^MID",
        "FTSE 100 (UK)": "^FTSE",
        "DAX (Germany)": "^GDAXI",
        "CAC 40 (France)": "^FCHI",
        "EURO STOXX 50": "^STOXX50E",
        "IBEX 35 (Spain)": "^IBEX",
        "Swiss Market Index (SMI)": "^SSMI",
        "TSX Composite (Canada)": "^GSPTSE",
        "IPC (Mexico)": "^MXX",
        "Bovespa (Brazil)": "^BVSP",
        "Merval (Argentina)": "^MERV",
        "Nikkei 225 (Japan)": "^N225",
        "Hang Seng (Hong Kong)": "^HSI",
        "Hang Seng China Enterprises": "^HSCE",
        "Shanghai Composite (Mainland China)": "000001.SS",
        "Shenzhen Composite (Mainland China)": "399001.SZ",
        "KOSPI Composite (South Korea)": "^KS11",
        "Straits Times Index (Singapore)": "^STI",
        "SENSEX (India)": "^BSESN",
        "NIFTY 50 (India)": "^NSEI",
        "Jakarta Composite (Indonesia)": "^JKSE",
        "Taiwan Weighted Index (Taiwan)": "^TWII",
        "S&P/ASX 200 (Australia)": "^AXJO"
    }
    
    # Convert to a list of strings in "Name - Ticker" format
    benchmark_list = [f"{name} - {ticker}" for name, ticker in benchmark_dict.items()]
    
    # Return the list and a None for error_message
    return benchmark_list, None


def fetch_benchmark_data(benchmark_ticker, start_date, end_date):
    """
    Fetch historical data for the selected benchmark symbol using Yahoo Finance (yfinance).
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


def calculate_drawdowns(portfolio_returns):
    """Calculate drawdowns over time."""
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return drawdown

def plot_drawdowns(drawdown, dates):
    """Plot drawdowns over time."""
    fig = px.line(x=dates, y=drawdown, title="Portfolio Drawdown Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown", showlegend=True)
    st.plotly_chart(fig)

def monte_carlo_simulation(portfolio_returns, num_simulations=10000, days=252):
    """Simulate future portfolio returns using Monte Carlo methods."""
    mean_return = np.mean(portfolio_returns)
    std_dev = np.std(portfolio_returns)
    simulated_returns = np.random.normal(mean_return, std_dev, (num_simulations, days))
    simulated_portfolio_values = portfolio_value * (1 + simulated_returns).cumprod(axis=1)
    return simulated_portfolio_values

def plot_monte_carlo_distribution(simulated_portfolio_values):
    """Plot the distribution of simulated portfolio values."""
    fig = px.histogram(simulated_portfolio_values[:, -1], nbins=50, title="Monte Carlo Simulation of Portfolio Value")
    fig.update_layout(xaxis_title="Portfolio Value", yaxis_title="Frequency")
    st.plotly_chart(fig)


def calculate_tail_risk(portfolio_returns, threshold_percentile=5):
    """Analyze tail risk using Extreme Value Theory (EVT)."""
    threshold = np.percentile(portfolio_returns, threshold_percentile)
    tail_returns = portfolio_returns[portfolio_returns < threshold]
    params = genpareto.fit(tail_returns)
    return params, threshold, tail_returns

def plot_tail_risk(tail_returns, threshold):
    """Plot the tail risk distribution."""
    fig = px.histogram(tail_returns, nbins=50, title="Tail Risk Distribution (Extreme Value Theory)")
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold:.4f}")
    fig.update_layout(xaxis_title="Returns", yaxis_title="Frequency")
    st.plotly_chart(fig)

def stress_test(portfolio_returns, stress_scenario):
    """Simulate extreme market scenarios and analyze their impact."""
    stressed_returns = portfolio_returns * stress_scenario
    stressed_cumulative_returns = np.cumprod(1 + stressed_returns) - 1
    return stressed_cumulative_returns

def plot_stress_test(stressed_cumulative_returns, dates):
    """Plot the impact of stress scenarios on the portfolio."""
    fig = px.line(x=dates, y=stressed_cumulative_returns, title="Portfolio Performance Under Stress Scenario")
    fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Returns", showlegend=True)
    st.plotly_chart(fig)

st.sidebar.header("Risk Analytics")
risk_analysis = st.sidebar.checkbox("Run Risk Analytics")
confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95) / 100
rolling_window = st.sidebar.number_input("Rolling Volatility Window (Days):", min_value=1, value=30)


benchmarks_list, error_message = fetch_benchmark()

if error_message:
    st.sidebar.error(error_message)
    benchmark_ticker = '^GSPC'  # fallback
else:
    # Let the user select from the list of "Name - Ticker" strings
    selected_benchmark = st.sidebar.selectbox("Select Benchmark Index:", options=benchmarks_list)
    # Extract ticker by splitting on " - "
    benchmark_ticker = selected_benchmark.split(" - ")[1]


if st.sidebar.button("Run Risk Analytics"):
    if 'all_returns' in st.session_state and 'stock_data_dict' in st.session_state:
        all_returns = st.session_state['all_returns']
        stock_data_dict = st.session_state['stock_data_dict']
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

        # Display risk metrics in a 3x3 grid
        st.write(f"### Portfolio Risk Metrics at {confidence_level * 100:.0f}% Confidence")
        
        # Row 1: VaR, CVaR, and Beta Analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Value at Risk (VaR)", value=f"{var:.4f}")
        with col2:
            st.metric(label="Conditional Value at Risk (CVaR)", value=f"{cvar:.4f}")
        with col3:
            benchmark_data, error_message = fetch_benchmark_data(benchmark_ticker, start_date.isoformat(), end_date.isoformat())
            if benchmark_data is not None:
                benchmark_returns = benchmark_data['returns'].dropna().values
                portfolio_returns_aligned = portfolio_returns[:len(benchmark_returns)]  # Align lengths
                beta = calculate_beta(portfolio_returns_aligned, benchmark_returns)
                st.metric(label=f"Beta (vs {selected_benchmark})", value=f"{beta:.4f}")
            else:
                st.error(error_message)

        # Row 2: Historical VaR, Variance-Covariance VaR, Monte Carlo VaR
        col1, col2, col3 = st.columns(3)
        with col1:
            historical_var = calculate_historical_var(portfolio_returns, confidence_level)
            st.metric(label="Historical VaR", value=f"{historical_var:.4f}")
        with col2:
            variance_covariance_var = calculate_variance_covariance_var(portfolio_returns, confidence_level)
            st.metric(label="Variance-Covariance VaR", value=f"{variance_covariance_var:.4f}")
        with col3:
            monte_carlo_var = calculate_monte_carlo_var(portfolio_returns, confidence_level)
            st.metric(label="Monte Carlo VaR", value=f"{monte_carlo_var:.4f}")

        # Row 3: Calmar Ratio, Maximum Drawdown, Expected Shortfall
        col1, col2, col3 = st.columns(3)
        with col1:
            calmar_ratio = calculate_calmar_ratio(portfolio_returns)
            st.metric(label="Calmar Ratio", value=f"{calmar_ratio:.4f}")
        with col2:
            max_drawdown = calculate_maximum_drawdown(portfolio_returns)
            st.metric(label="Maximum Drawdown", value=f"{max_drawdown:.4f}")
        with col3:
            expected_shortfall = calculate_expected_shortfall(portfolio_returns, confidence_level)
            st.metric(label="Expected Shortfall", value=f"{expected_shortfall:.4f}")

        # Plot portfolio return distribution with VaR methods
        st.write("### Portfolio Return Distribution with VaR Methods")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=portfolio_returns, nbinsx=50, name="Portfolio Returns"))
        
        # Add vertical lines for VaR methods
        fig.add_vline(x=-historical_var, line_dash="dash", line_color="red", annotation_text="Historical VaR", annotation_position="top")
        fig.add_vline(x=-variance_covariance_var, line_dash="dash", line_color="blue", annotation_text="Variance-Covariance VaR", annotation_position="top")
        fig.add_vline(x=-monte_carlo_var, line_dash="dash", line_color="green", annotation_text="Monte Carlo VaR", annotation_position="top")
        
        fig.update_layout(
            title="Portfolio Return Distribution with VaR Methods",
            xaxis_title="Returns",
            yaxis_title="Frequency",
            showlegend=True
        )
        st.plotly_chart(fig)

        # Rolling Volatility
        rolling_volatility = calculate_rolling_volatility(pd.Series(portfolio_returns), rolling_window)
        st.write("### Rolling Volatility")
        fig = px.line(rolling_volatility, title="Rolling Volatility")
        st.plotly_chart(fig)

        # 🔹 New: Drawdown Analysis
        st.write("### Drawdown Analysis")
        drawdown = calculate_drawdowns(portfolio_returns)
        dates = stock_data_dict[stock_names[0]]['date'][1:len(drawdown)+1]  # Align dates with drawdowns
        plot_drawdowns(drawdown, dates)

        # 🔹 New: Monte Carlo Simulation
        st.write("### Monte Carlo Simulation of Future Portfolio Returns")
        simulated_portfolio_values = monte_carlo_simulation(portfolio_returns)
        plot_monte_carlo_distribution(simulated_portfolio_values)

        # 🔹 New: Tail Risk Analysis (EVT)
        st.write("### Tail Risk Analysis (Extreme Value Theory)")
        params, threshold, tail_returns = calculate_tail_risk(portfolio_returns)
        st.write(f"Tail Risk Parameters (GPD): {params}")
        plot_tail_risk(tail_returns, threshold)

        # 🔹 New: Stress Testing
        st.write("### Stress Testing (Market Crash Scenario)")
        stress_scenario = np.random.choice([0.9, 0.8, 0.7], size=len(portfolio_returns))  # Simulate a 10-30% drop
        stressed_cumulative_returns = stress_test(portfolio_returns, stress_scenario)
        plot_stress_test(stressed_cumulative_returns, dates)

    else:
        st.error("Please fetch data first.")

st.sidebar.title("Connect with Me")
github_icon = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
linkedin_icon = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
email_icon = "https://cdn-icons-png.flaticon.com/512/732/732200.png"

# Using st.markdown with HTML to include hyperlinks with images
st.sidebar.markdown(
    f"""
    <div style="display: flex; justify-content: space-evenly; align-items: center;">
        <a href="https://github.com/kushalgowdagv" target="_blank">
            <img src="{github_icon}" width="30">
        </a>
        <a href="mailto:kushalgowdagv@gmail.com" target="_blank">
            <img src="{email_icon}" width="30">
        </a>
        <a href="https://www.linkedin.com/in/kushalgowdagv/" target="_blank">
            <img src="{linkedin_icon}" width="30">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# ... (rest of your existing code)

# Footer Section
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <h1 style="text-align: center; margin-top: 20px;">Contact Me</h1>

    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
        <a href="https://github.com/kushalgowdagv" target="_blank" style="margin-right: 15px;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="40" title="GitHub">
        </a>
        <a href="mailto:kushalgowdagv@gmail.com" target="_blank" style="margin-right: 15px;">
            <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="40" title="Email">
        </a>
        <a href="https://www.linkedin.com/in/kushalgowdagv/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40" title="LinkedIn">
        </a>
    </div>
    <p style="text-align: center; font-size: 16px; margin-top: 10px;">Feel free to connect with me on any of the platforms above!</p>
    """,
    unsafe_allow_html=True
)