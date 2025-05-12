import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
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