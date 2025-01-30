# Multi-Dimensional Portfolio Risk and Performance Analysis Tool

## Overview
This Streamlit-based application is designed to provide a comprehensive analysis of investment portfolios. It allows users to input their portfolio details, fetch historical stock data, and analyze various risk and performance metrics. The tool also includes portfolio optimization and advanced risk analytics features.

## Features
- **Portfolio Configuration**: Users can input the portfolio value, select the number of stocks, and specify the tickers and weights.
- **Historical Data Fetching**: The tool fetches historical stock data for the specified date range.
- **Performance Metrics**: Calculates key metrics such as CAGR, Volatility, and Sharpe Ratio for each stock and the overall portfolio.
- **Correlation Heatmap**: Visualizes the correlation between the returns of different stocks in the portfolio.
- **Portfolio Optimization**: Offers optimization strategies to maximize Sharpe Ratio, minimize volatility, or maximize return.
- **Risk Analytics**: Provides detailed risk metrics including Value at Risk (VaR), Conditional Value at Risk (CVaR), and more.
- **Monte Carlo Simulation**: Simulates future portfolio returns to assess potential outcomes.
- **Stress Testing**: Analyzes the impact of extreme market scenarios on the portfolio.

## How to Use
1. **Portfolio Configuration**:
   - Enter the total portfolio value.
   - Select the number of stocks and input their tickers and weights.
   - Ensure the total weight sums to 100%.

2. **Date Range**:
   - Select the start and end dates for the analysis.

3. **Fetch Data**:
   - Click the "Fetch Data" button to retrieve historical stock data.

4. **Performance Analysis**:
   - View price movement charts, cumulative portfolio returns, and correlation heatmaps.
   - Review key metrics such as CAGR, Volatility, and Sharpe Ratio.

5. **Portfolio Optimization (Optional)**:
   - Check the "Optimize Portfolio" checkbox and select an optimization strategy.
   - Click "Run Optimization" to view optimized weights and performance.

6. **Risk Analytics (Optional)**:
   - Check the "Run Risk Analytics" checkbox and set the confidence level and rolling volatility window.
   - Click "Run Risk Analytics" to view detailed risk metrics and simulations.

## Example Walkthrough
1. **Portfolio Configuration**:
   - Portfolio Value: $100,000
   - Number of Stocks: 2
   - Stock Tickers: AAPL (60%), MSFT (40%)

2. **Date Range**:
   - Start Date: 2022-01-01
   - End Date: 2023-01-01

3. **Fetch Data**:
   - Click "Fetch Data" to load historical data and view initial metrics.

4. **Optimization**:
   - Check "Optimize Portfolio" and select "Max Sharpe Ratio".
   - Click "Run Optimization" to view optimized weights and performance.

5. **Risk Analytics**:
   - Check "Run Risk Analytics", set Confidence Level to 95%, and select Benchmark Index ^GSPC.
   - Click "Run Risk Analytics" to view detailed risk metrics and simulations.

## Key Notes
- If "Optimize Portfolio" is checked, the risk metrics will use the optimized weights. Otherwise, the initial weights will be used.
- The introduction will disappear once you click "Fetch Data".

## Dependencies
- Streamlit
- NumPy
- Pandas
- Requests
- Plotly
- SciPy
- yfinance

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kushalgowdagv/portfolio-risk-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd portfolio-risk-analysis
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Connect with Me
Feel free to connect with me on any of the platforms below:
- [GitHub](https://github.com/kushalgowdagv)
- [Email](mailto:kushalgowdagv@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/kushalgowdagv/)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy analyzing! 📊📈