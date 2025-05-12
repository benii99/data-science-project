import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates

def temporal_pattern_analysis(processed_dfs, output_dir="figures/"):
    """
    Perform temporal pattern analysis on air quality data.
    
    Parameters:
    processed_dfs: Dictionary of location names and processed DataFrames
    output_dir: Directory to save output figures
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for location, df in processed_dfs.items():
        print(f"Performing temporal pattern analysis for {location}...")
        
        # Ensure data is sorted by time
        df = df.sort_values('time')
        df = df.set_index('time')
        
        # Get AQI time series with no missing values
        aqi_series = df['AQI'].dropna()
        
        # 1. Time Series Decomposition using STL
        stl = STL(aqi_series, seasonal=25, period=24, robust=True)
        result = stl.fit()
        
        # Plot STL Decomposition
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(result.observed)
        axes[0].set_ylabel('Observed')
        axes[0].set_title(f'STL Decomposition of AQI for {location}')
        
        axes[1].plot(result.trend)
        axes[1].set_ylabel('Trend')
        
        axes[2].plot(result.seasonal)
        axes[2].set_ylabel('Daily\nSeasonality')
        
        axes[3].plot(result.resid)
        axes[3].set_ylabel('Residuals')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{location.replace(' ', '_')}_stl_decomposition.png", dpi=300)
        
        # 2. Daily and Weekly Patterns
        # Extract hour of day and day of week
        df_hourly = df.reset_index()
        df_hourly['hour'] = df_hourly['time'].dt.hour
        df_hourly['day_of_week'] = df_hourly['time'].dt.dayofweek
        df_hourly['day_name'] = df_hourly['time'].dt.day_name()
        
        # Heatmap of hour vs day of week
        pivot_table = df_hourly.pivot_table(
            values='AQI', 
            index='hour',
            columns='day_name', 
            aggfunc='mean'
        )
        
        # Reorder days of week
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table[days_order]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.1f')
        plt.title(f'Average AQI by Hour and Day of Week - {location}')
        plt.ylabel('Hour of Day')
        plt.xlabel('Day of Week')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{location.replace(' ', '_')}_hourly_weekly_heatmap.png", dpi=300)
        
        # 3. Monthly and Seasonal Patterns
        df_monthly = df.reset_index()
        df_monthly['month'] = df_monthly['time'].dt.month
        df_monthly['month_name'] = df_monthly['time'].dt.month_name()
        
        # Monthly boxplots
        plt.figure(figsize=(14, 6))
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                         'July', 'August', 'September', 'October', 'November', 'December']
        
        # Filter to keep only month names that exist in the data
        available_months = df_monthly['month_name'].unique()
        ordered_months = [m for m in months_order if m in available_months]
        
        # Create boxplot with available months
        ax = sns.boxplot(x='month_name', y='AQI', data=df_monthly, 
                         order=ordered_months, palette='viridis')
        plt.title(f'Monthly Distribution of AQI - {location}')
        plt.xlabel('Month')
        plt.ylabel('AQI Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{location.replace(' ', '_')}_monthly_boxplot.png", dpi=300)
        
        # 4. Autocorrelation Analysis
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF for up to 7 days (7*24 hours) with 95% confidence intervals
        plot_acf(aqi_series, lags=7*24, ax=axes[0], alpha=0.05)
        axes[0].set_title(f'Autocorrelation Function (ACF) - {location}')
        axes[0].set_xlabel('Lag (hours)')
        
        # PACF
        plot_pacf(aqi_series, lags=7*24, ax=axes[1], alpha=0.05)
        axes[1].set_title(f'Partial Autocorrelation Function (PACF) - {location}')
        axes[1].set_xlabel('Lag (hours)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{location.replace(' ', '_')}_acf_pacf.png", dpi=300)
        
        # 5. Rolling statistics
        rolling_window = 24*7  # One week
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Calculate rolling mean and standard deviation
        rolling_mean = aqi_series.rolling(window=rolling_window).mean()
        rolling_std = aqi_series.rolling(window=rolling_window).std()
        
        # Plot the rolling statistics
        ax.plot(aqi_series, alpha=0.5, label='Original Data')
        ax.plot(rolling_mean, linewidth=2, label=f'{rolling_window}-hour Rolling Mean')
        ax.plot(rolling_mean + 2*rolling_std, linestyle='--', label='Upper Band (+2σ)')
        ax.plot(rolling_mean - 2*rolling_std, linestyle='--', label='Lower Band (-2σ)')
        
        # Format x-axis to show months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        
        ax.set_title(f'Weekly Rolling Statistics of AQI - {location}')
        ax.set_xlabel('Date')
        ax.set_ylabel('AQI Value')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{location.replace(' ', '_')}_rolling_stats.png", dpi=300)
        
        print(f"Completed temporal analysis for {location}")

def spatial_comparison_analysis(processed_dfs, output_dir="figures/"):
    """
    Perform spatial comparison analysis between different locations.
    
    Parameters:
    processed_dfs: Dictionary of location names and processed DataFrames
    output_dir: Directory to save output figures
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prepare data for comparison
    location_series = {}
    for location, df in processed_dfs.items():
        # Ensure data is sorted by time and set index
        df = df.sort_values('time')
        location_series[location] = df.set_index('time')['AQI']
    
    # Create a combined DataFrame
    combined_df = pd.DataFrame(location_series)
    combined_df = combined_df.dropna()
    
    # 2. Distribution comparison
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=combined_df, palette="Set3")
    plt.title('AQI Distribution Comparison Across Locations')
    plt.ylabel('AQI Value')
    plt.xlabel('Location')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/location_aqi_distribution.png", dpi=300)
    
    # 3. Time series comparison
    plt.figure(figsize=(14, 8))
    for location, series in location_series.items():
        plt.plot(series, label=location, alpha=0.7)
    
    plt.title('AQI Time Series Comparison Across Locations')
    plt.xlabel('Date')
    plt.ylabel('AQI Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/location_time_series_comparison.png", dpi=300)
    
    # 4. Correlation analysis
    correlation = combined_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Between Locations')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/location_correlation.png", dpi=300)
    
    # 5. Cross-correlation analysis with lags
    max_lags = 24  # 24 hours lag
    lags = range(-max_lags, max_lags + 1)
    locations = list(location_series.keys())
    n_locations = len(locations)
    
    # Create a grid of cross-correlation plots
    fig, axes = plt.subplots(n_locations, n_locations, figsize=(15, 15), sharex=True, sharey=True)
    
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            # Calculate cross-correlation
            s1 = location_series[loc1].values
            s2 = location_series[loc2].values
            
            # Ensure series have the same length
            min_len = min(len(s1), len(s2))
            s1 = s1[:min_len]
            s2 = s2[:min_len]
            
            # Calculate cross-correlation
            xcorr = [np.corrcoef(s1[max(0, -k):min(min_len, min_len-k)], 
                                s2[max(0, k):min(min_len, min_len+k)])[0, 1] 
                    for k in lags]
            
            # Plot cross-correlation
            axes[i, j].plot(lags, xcorr)
            axes[i, j].axvline(x=0, color='r', linestyle='--', alpha=0.3)
            axes[i, j].set_title(f'{loc1} vs {loc2}', fontsize=10)
            
            # Add grid
            axes[i, j].grid(True, alpha=0.3)
            
            # Only add x and y labels for outer plots
            if i == n_locations - 1:
                axes[i, j].set_xlabel('Lag (hours)')
            if j == 0:
                axes[i, j].set_ylabel('Correlation')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/location_cross_correlation.png", dpi=300)
    
    # 6. Compare daily patterns between locations
    daily_patterns = {}
    for location, df in processed_dfs.items():
        hourly_avg = df.groupby(df['time'].dt.hour)['AQI'].mean()
        daily_patterns[location] = hourly_avg
    
    daily_patterns_df = pd.DataFrame(daily_patterns)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_patterns_df)
    plt.title('Average Daily AQI Pattern by Location')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average AQI')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/location_daily_patterns.png", dpi=300)
    
    print("Completed spatial comparison analysis")

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
