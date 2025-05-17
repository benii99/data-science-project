import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
from datetime import datetime

def traffic_aqi_correlation_analysis(df, output_dir="figures/traffic_aqi"):
    """
    Perform correlation analysis between traffic data and AQI.
    
    Parameters:
    df: DataFrame containing both traffic and AQI data
    output_dir: Directory to save output figures
    
    Returns:
    dict with correlation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date for filename
    date_str = datetime.now().strftime("%Y%m%d")
    
    print("\n--- Traffic-AQI Correlation Analysis ---")
    
    # Make a clean copy of the dataframe
    df_clean = df.copy()
    
    # Ensure columns exist and have proper names
    required_columns = ['traffic_count', 'AQI']
    if 'datetime' in df_clean.columns and 'time' not in df_clean.columns:
        df_clean = df_clean.rename(columns={'datetime': 'time'})
        
    for col in required_columns:
        if col not in df_clean.columns:
            print(f"Error: Required column '{col}' not found in dataset")
            return None
    
    # Check data types and convert if necessary
    print("Converting data to numeric types...")
    df_clean['traffic_count'] = pd.to_numeric(df_clean['traffic_count'], errors='coerce')
    df_clean['AQI'] = pd.to_numeric(df_clean['AQI'], errors='coerce')
    
    # Drop any rows with NaN values
    orig_len = len(df_clean)
    df_clean = df_clean.dropna(subset=['traffic_count', 'AQI'])
    dropped = orig_len - len(df_clean)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaN values")
    
    print(f"Working with {len(df_clean)} valid data points")
    
    # Print summary to validate the data
    print("\nTraffic count range:", df_clean['traffic_count'].min(), "to", df_clean['traffic_count'].max())
    print("AQI range:", df_clean['AQI'].min(), "to", df_clean['AQI'].max())
    
    # Try pandas correlation first (more robust than scipy for large datasets)
    print("\nCalculating correlation using pandas method...")
    pearson_corr_pd = df_clean['traffic_count'].corr(df_clean['AQI'], method='pearson')
    spearman_corr_pd = df_clean['traffic_count'].corr(df_clean['AQI'], method='spearman')
    
    print(f"Pandas Pearson correlation: {pearson_corr_pd:.4f}")
    print(f"Pandas Spearman correlation: {spearman_corr_pd:.4f}")
    
    # Try scipy correlation with explicit error handling
    print("\nAttempting scipy correlation calculation...")
    try:
        # Convert to numpy arrays explicitly
        traffic_array = df_clean['traffic_count'].values
        aqi_array = df_clean['AQI'].values
        
        pearson_corr, pearson_p = pearsonr(traffic_array, aqi_array)
        spearman_corr, spearman_p = spearmanr(traffic_array, aqi_array)
        
        print(f"Successfully calculated correlations with scipy")
    except Exception as e:
        print(f"Error with scipy correlation: {e}")
        # Use pandas results as fallback
        pearson_corr, pearson_p = pearson_corr_pd, np.nan
        spearman_corr, spearman_p = spearman_corr_pd, np.nan
    
    print(f"\nCorrelation between Traffic Count and AQI:")
    print(f"Pearson correlation: {pearson_corr:.4f}" + (f" (p-value: {pearson_p:.4f})" if not np.isnan(pearson_p) else ""))
    print(f"Spearman correlation: {spearman_corr:.4f}" + (f" (p-value: {spearman_p:.4f})" if not np.isnan(spearman_p) else ""))
    
    # Create scatter plot of traffic count vs AQI
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='traffic_count', y='AQI', data=df_clean, alpha=0.3)
    
    # Add regression line
    sns.regplot(x='traffic_count', y='AQI', data=df_clean, scatter=False, ci=None, line_kws={"color": "red"})
    
    plt.title(f'Traffic Count vs Air Quality Index (AQI)\nPearson r = {pearson_corr:.4f}')
    plt.xlabel('Traffic Count')
    plt.ylabel('Air Quality Index (AQI)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_traffic_aqi_scatter.png", dpi=300)
    plt.close()
    
    # Add date components for analysis
    if 'time' in df_clean.columns:
        df_clean['hour'] = df_clean['time'].dt.hour
        df_clean['day_of_week'] = df_clean['time'].dt.dayofweek
        df_clean['month'] = df_clean['time'].dt.month
        
        # Create daily average plot
        plt.figure(figsize=(12, 6))
        daily_data = df_clean.groupby(df_clean['time'].dt.date).agg({
            'traffic_count': 'mean',
            'AQI': 'mean'
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Traffic Count', color=color)
        ax1.plot(daily_data['time'], daily_data['traffic_count'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('AQI', color=color)
        ax2.plot(daily_data['time'], daily_data['AQI'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Daily Average Traffic Count and AQI')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{date_str}_daily_traffic_aqi.png", dpi=300)
        plt.close()
        
        # Create hourly pattern plot
        plt.figure(figsize=(12, 6))
        hourly_data = df_clean.groupby('hour').agg({
            'traffic_count': 'mean',
            'AQI': 'mean'
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Traffic Count', color=color)
        ax1.plot(hourly_data['hour'], hourly_data['traffic_count'], 'o-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Average AQI', color=color)
        ax2.plot(hourly_data['hour'], hourly_data['AQI'], 'o-', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Hourly Pattern of Traffic Count and AQI')
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{date_str}_hourly_pattern.png", dpi=300)
        plt.close()
    
    # Return correlation results
    return {
        'pearson': (pearson_corr, pearson_p if not np.isnan(pearson_p) else None),
        'spearman': (spearman_corr, spearman_p if not np.isnan(spearman_p) else None),
        'pearson_pd': pearson_corr_pd,
        'spearman_pd': spearman_corr_pd
    }
