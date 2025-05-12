import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

def sarima_modeling(processed_dfs, output_dir="figures/"):
    """
    Perform SARIMA modeling and forecasting for each location.
    
    Parameters:
    processed_dfs: Dictionary of location names and processed DataFrames
    output_dir: Directory to save output figures
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Model configuration
    # SARIMA order (p,d,q)(P,D,Q,s) where s is the seasonal period
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 24)  # 24-hour seasonality
    
    for location, df in processed_dfs.items():
        print(f"Performing SARIMA modeling for {location}...")
        
        # Prepare data
        df = df.sort_values('time')
        df = df.set_index('time')
        series = df['AQI'].dropna()
        
        # Split data into train and test sets
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        
        # Fit SARIMA model
        model = SARIMAX(train, 
                        order=order, 
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        fitted_model = model.fit(disp=False)
        print(f"Model summary for {location}:")
        print(fitted_model.summary())
        
        # Make predictions on test set
        forecast = fitted_model.get_forecast(steps=len(test))
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Calculate metrics
        mse = mean_squared_error(test, forecast_mean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, forecast_mean)
        
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # Plot the forecast vs actual
        plt.figure(figsize=(14, 7))
        plt.plot(series, label='Observed', alpha=0.7)
        plt.plot(forecast_mean, color='r', label='Forecast')
        plt.fill_between(forecast_ci.index,
                        forecast_ci.iloc[:, 0],
                        forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
        
        plt.axvline(train.index[-1], color='k', linestyle='--')
        plt.legend()
        plt.title(f'SARIMA Forecast for {location}\nRMSE: {rmse:.2f}, MAE: {mae:.2f}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{location.replace(' ', '_')}_sarima_forecast.png", dpi=300)
        
        # Make a 7-day future forecast
        future_steps = 24 * 7  # 7 days of hourly forecasts
        future_forecast = fitted_model.get_forecast(steps=future_steps)
        future_mean = future_forecast.predicted_mean
        future_ci = future_forecast.conf_int()
        
        # Plot future forecast
        plt.figure(figsize=(14, 7))
        plt.plot(series[-30*24:], label='Recent Observations', alpha=0.7)  # Last 30 days
        plt.plot(future_mean, color='r', label='7-Day Forecast')
        plt.fill_between(future_ci.index,
                        future_ci.iloc[:, 0],
                        future_ci.iloc[:, 1], color='pink', alpha=0.3)
        
        plt.legend()
        plt.title(f'7-Day AQI Forecast for {location}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{location.replace(' ', '_')}_future_forecast.png", dpi=300)
        
        print(f"Completed SARIMA modeling for {location}")
