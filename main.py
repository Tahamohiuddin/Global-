from data_processing import load_data, preprocess_data
from visualization import plot_interactive_trends, plot_geographical_heatmap
from analysis import perform_regression_analysis, arima_forecast, interpret_results

def main():
    # Load and preprocess data
    print("Loading data...")
    url = 'https://datahub.io/core/global-temp/r/monthly.csv'
    df = load_data(url)
    annual_avg = preprocess_data(df)

    # Visualize temperature trends
    print("Generating visualizations...")
    plot_interactive_trends(annual_avg)
    
    # Perform regression and ARIMA forecasting
    print("Performing trend analysis and forecasting...")
    perform_regression_analysis(annual_avg)
    arima_forecast(annual_avg, forecast_years=50)

    # Geographical heatmap for regional temperatures (dummy data)
    plot_geographical_heatmap()

    # Interpret findings
    interpret_results(annual_avg)

if __name__ == "__main__":
    main()
