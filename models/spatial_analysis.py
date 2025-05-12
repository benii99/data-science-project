import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

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