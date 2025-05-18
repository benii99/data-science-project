import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd
import os

def temporal_pattern_analysis(processed_dfs, output_dir="figures/"):
    """
    Perform temporal pattern analysis on air quality data.
    Creates ONE multi-panel figure showing daily/weekly/seasonal patterns
    focused on primary location (Torvegade) and AQI + ozone + nitrogen_dioxide.
    
    Parameters:
    processed_dfs: Dictionary of location names and processed DataFrames
    output_dir: Directory to save output figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Primary location to focus on
    primary_location = "Torvegade"
    
    if primary_location not in processed_dfs:
        print(f"Error: {primary_location} not found in processed data")
        return
    
    print(f"Performing temporal pattern analysis for {primary_location}...")
    
    # Get data for the primary location
    df = processed_dfs[primary_location].copy()
    
    # Check if required columns exist
    required_columns = ['time', 'AQI', 'ozone', 'nitrogen_dioxide']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {', '.join(missing_columns)}")
        # Keep only available columns
        available_columns = [col for col in required_columns if col in df.columns]
        if 'time' not in available_columns:
            print("Error: 'time' column is required")
            return
    else:
        available_columns = required_columns
    
    # Filter to keep only necessary columns
    df = df[available_columns]
    
    # Ensure time is datetime and sort
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # Create one multi-panel figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    
    # 1. Daily Patterns (by hour of day)
    df['hour'] = df['time'].dt.hour
    daily_patterns = df.groupby('hour').mean(numeric_only=True)
    
    # Plot daily patterns for each available pollutant/AQI
    for col in [col for col in ['AQI', 'ozone', 'nitrogen_dioxide'] if col in df.columns]:
        axes[0].plot(daily_patterns.index, daily_patterns[col], marker='o', label=col)
    
    axes[0].set_title(f'Daily Patterns (by Hour) - {primary_location}')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Value')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Weekly Patterns (by day of week)
    df['day_of_week'] = df['time'].dt.dayofweek
    weekly_patterns = df.groupby('day_of_week').mean(numeric_only=True)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Plot weekly patterns for each available pollutant/AQI
    for col in [col for col in ['AQI', 'ozone', 'nitrogen_dioxide'] if col in df.columns]:
        axes[1].plot(range(len(days)), weekly_patterns[col], marker='o', label=col)
    
    axes[1].set_title(f'Weekly Patterns (by Day) - {primary_location}')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Value')
    axes[1].set_xticks(range(len(days)))
    axes[1].set_xticklabels(days)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Monthly/Seasonal Patterns
    df['month'] = df['time'].dt.month
    monthly_patterns = df.groupby('month').mean(numeric_only=True)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot monthly patterns for each available pollutant/AQI
    for col in [col for col in ['AQI', 'ozone', 'nitrogen_dioxide'] if col in df.columns]:
        axes[2].plot(range(1, len(months)+1), monthly_patterns[col], marker='o', label=col)
    
    axes[2].set_title(f'Monthly/Seasonal Patterns - {primary_location}')
    axes[2].set_xlabel('Month')
    axes[2].set_ylabel('Value')
    axes[2].set_xticks(range(1, len(months)+1))
    axes[2].set_xticklabels(months)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"{output_dir}/{primary_location.replace(' ', '_')}_temporal_patterns_combined.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved multi-panel temporal pattern figure to {output_file}")
