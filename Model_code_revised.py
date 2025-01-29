

import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.express as px
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
        weight = st.number_input(f"Weight (%) {i + 1}:", min_value=0.0, max_value=100.0, step=1.0, key=f"weight_{i}")
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
        portfolio_cumulative_returns, df_returns, metrics_data = calculate_portfolio_metrics(stock_data_dict, all_returns, min_length, stock_names, weights, start_date, end_date)

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
