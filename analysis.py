from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

def perform_regression_analysis(annual_avg):
    """Performs linear regression and plots the trend line."""
    X = annual_avg['Year'].values.reshape(-1, 1)
    y = annual_avg['Mean'].values

    reg = LinearRegression()
    reg.fit(X, y)
    trend_line = reg.predict(X)

    plt.figure(figsize=(12, 6))
    plt.plot(annual_avg['Year'], y, label='Temperature Anomaly', color='coral')
    plt.plot(annual_avg['Year'], trend_line, label=f'Linear Trendline (Slope: {reg.coef_[0]:.4f})', color='blue')
    plt.title('Linear Regression Analysis of Temperature Trends')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (°C)')
    plt.legend()
    plt.show()

def arima_forecast(annual_avg, forecast_years=50):
    """Performs ARIMA forecasting and plots future temperature trends."""
    y = annual_avg['Mean'].values
    arima_model = ARIMA(y, order=(5, 1, 0))
    arima_fit = arima_model.fit()

    future_years = np.arange(annual_avg['Year'].max() + 1, annual_avg['Year'].max() + 1 + forecast_years)
    forecast = arima_fit.forecast(steps=forecast_years)

    plt.figure(figsize=(12, 6))
    plt.plot(annual_avg['Year'], y, label='Historical Data')
    plt.plot(future_years, forecast, label='ARIMA Forecast', color='red')
    plt.title(f'Forecast of Global Temperatures (Next {forecast_years} Years)')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (°C)')
    plt.legend()
    plt.show()

def interpret_results(annual_avg):
    """Analyzes and prints a summary of the findings."""
    latest_year = annual_avg['Year'].max()
    latest_temp = annual_avg[annual_avg['Year'] == latest_year]['Mean'].values[0]

    if latest_temp >= 2.0:
        print(f"WARNING: The current global temperature anomaly in {latest_year} is {latest_temp:.2f}°C. Immediate action is required.")
    else:
        print(f"The global temperature anomaly in {latest_year} is {latest_temp:.2f}°C. Climate policies must remain effective to prevent a crisis.")
