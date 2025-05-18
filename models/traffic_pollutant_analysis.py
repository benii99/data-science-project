# models/traffic_pollutant_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
from datetime import datetime
from config import POLLUTANTS

def traffic_pollutant_correlation_analysis(df, output_dir="figures/traffic_pollutants"):
    """
    Perform correlation analysis between traffic data and individual pollutants.
    
    Parameters:
    df: DataFrame containing both traffic and pollutant data
    output_dir: Directory to save output figures
    
    Returns:
    dict with correlation results for each pollutant
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date for filename
    date_str = datetime.now().strftime("%Y%m%d")
    
    print("\n" + "="*80)
    print("TRAFFIC-POLLUTANT CORRELATION ANALYSIS")
    print("="*80)
    
    # Make a clean copy of the dataframe
    df_clean = df.copy()
    
    # Ensure time column exists with proper name
    if 'datetime' in df_clean.columns and 'time' not in df_clean.columns:
        df_clean = df_clean.rename(columns={'datetime': 'time'})
    
    # Ensure traffic_count column exists
    if 'traffic_count' not in df_clean.columns:
        print(f"Error: Required column 'traffic_count' not found in dataset")
        return None
    
    # Convert traffic_count to numeric
    df_clean['traffic_count'] = pd.to_numeric(df_clean['traffic_count'], errors='coerce')
    
    # Extract pollutant list from POLLUTANTS in config
    pollutant_list = POLLUTANTS.split(',') if isinstance(POLLUTANTS, str) else POLLUTANTS
    
    # Add AQI to the list for comparison
    all_targets = pollutant_list + ['AQI']
    
    # Check which pollutants are available in the data
    available_pollutants = [p for p in all_targets if p in df_clean.columns]
    
    if not available_pollutants:
        print("No pollutant columns found in the dataset")
        return None
    
    print(f"Found {len(available_pollutants)} pollutants in dataset: {', '.join(available_pollutants)}")
    
    # Create a comprehensive correlation table for all pollutants vs traffic
    pollutant_correlations = {}
    pollutant_p_values = {}
    
    # Initialize a DataFrame to store all correlations
    correlation_matrix = pd.DataFrame(index=['traffic_count'], columns=available_pollutants)
    
    print("\n--- Calculating correlations between traffic and each pollutant ---")
    for pollutant in available_pollutants:
        # Convert pollutant to numeric
        df_clean[pollutant] = pd.to_numeric(df_clean[pollutant], errors='coerce')
        
        # Drop NaN values for this specific pollutant
        valid_data = df_clean.dropna(subset=['traffic_count', pollutant])
        
        if len(valid_data) < 10:
            print(f"Not enough valid data points for {pollutant}")
            correlation_matrix.loc['traffic_count', pollutant] = np.nan
            pollutant_correlations[pollutant] = np.nan
            pollutant_p_values[pollutant] = np.nan
            continue
        
        # Calculate correlation
        try:
            corr, p_value = pearsonr(valid_data['traffic_count'], valid_data[pollutant])
            correlation_matrix.loc['traffic_count', pollutant] = corr
            pollutant_correlations[pollutant] = corr
            pollutant_p_values[pollutant] = p_value
            print(f"{pollutant}: r = {corr:.4f} (p-value: {p_value:.4f})")
        except Exception as e:
            print(f"Error calculating correlation for {pollutant}: {e}")
            correlation_matrix.loc['traffic_count', pollutant] = np.nan
            pollutant_correlations[pollutant] = np.nan
            pollutant_p_values[pollutant] = np.nan
    
    # Save correlation table to CSV
    correlation_matrix.to_csv(os.path.join(output_dir, "traffic_pollutant_correlations.csv"))
    
    # Create bar chart of correlations - ONE comparative bar chart
    plt.figure(figsize=(12, 6))
    correlation_values = [pollutant_correlations[p] for p in available_pollutants]
    bars = plt.bar(available_pollutants, correlation_values, color=['green' if x > 0 else 'red' for x in correlation_values])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.02 if height >= 0 else height - 0.08,
                f'{height:.3f}',
                ha='center', 
                va='bottom' if height >= 0 else 'top',
                fontweight='bold'
            )
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Correlation between Traffic Count and Air Pollutants')
    plt.xlabel('Pollutant')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{date_str}_traffic_pollutant_correlations.png"), dpi=300)
    plt.close()
    
    # Find top 2 pollutants with strongest correlations (by absolute value)
    pollutants_abs_corr = {p: abs(pollutant_correlations[p]) for p in pollutant_correlations}
    top_pollutants = sorted(pollutants_abs_corr.items(), key=lambda x: x[1], reverse=True)
    
    # Always include AQI in detailed analysis if available
    detailed_analysis_pollutants = []
    if 'AQI' in available_pollutants:
        detailed_analysis_pollutants.append('AQI')
    
    # Add top 2 pollutants (excluding AQI if it's already there)
    for pollutant, corr in top_pollutants:
        if pollutant != 'AQI' and len(detailed_analysis_pollutants) < 4:  # Limit to 4 total (AQI + top 3)
            detailed_analysis_pollutants.append(pollutant)
    
    print(f"\nPerforming detailed analysis for: {', '.join(detailed_analysis_pollutants)}")
    
    # Detailed analysis for selected pollutants
    for pollutant in detailed_analysis_pollutants:
        print(f"\n--- Detailed Analysis for {pollutant} ---")
        
        # Create scatter plot with regression line
        plt.figure(figsize=(10, 6))
        valid_data = df_clean.dropna(subset=['traffic_count', pollutant])
        
        sns.scatterplot(x='traffic_count', y=pollutant, data=valid_data, alpha=0.3)
        sns.regplot(x='traffic_count', y=pollutant, data=valid_data, scatter=False, line_kws={"color": "red"})
        
        plt.title(f'Traffic Count vs {pollutant}\nPearson r = {pollutant_correlations[pollutant]:.4f}')
        plt.xlabel('Traffic Count')
        plt.ylabel(pollutant)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{date_str}_{pollutant}_scatter.png"), dpi=300)
        plt.close()
        
        # Create time series plot if time column exists
        if 'time' in valid_data.columns:
            # Daily pattern
            plt.figure(figsize=(12, 6))
            daily_data = valid_data.groupby(valid_data['time'].dt.date).agg({
                'traffic_count': 'mean',
                pollutant: 'mean'
            }).reset_index()
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Traffic Count', color=color)
            ax1.plot(daily_data['time'], daily_data['traffic_count'], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel(pollutant, color=color)
            ax2.plot(daily_data['time'], daily_data[pollutant], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title(f'Daily Average Traffic Count and {pollutant}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{date_str}_{pollutant}_daily.png"), dpi=300)
            plt.close()
            
            # Add hour of day component for hourly analysis
            valid_data['hour'] = valid_data['time'].dt.hour
            
            # Hourly pattern plot
            hourly_data = valid_data.groupby('hour').agg({
                'traffic_count': 'mean',
                pollutant: 'mean'
            }).reset_index()
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            color = 'tab:blue'
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Average Traffic Count', color=color)
            ax1.plot(hourly_data['hour'], hourly_data['traffic_count'], 'o-', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel(f'Average {pollutant}', color=color)
            ax2.plot(hourly_data['hour'], hourly_data[pollutant], 'o-', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title(f'Hourly Pattern of Traffic Count and {pollutant}')
            plt.xticks(range(0, 24))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{date_str}_{pollutant}_hourly.png"), dpi=300)
            plt.close()
            
            # Lagged correlation analysis
            max_lag_hours = 12  # Maximum lag to consider
            lags = range(-max_lag_hours, max_lag_hours + 1)
            correlations = []
            
            print(f"Calculating lagged correlations for {pollutant}...")
            
            for lag in lags:
                # Shift traffic data by lag
                lagged_traffic = valid_data['traffic_count'].shift(lag)
                
                # Calculate correlation between lagged traffic and pollutant
                valid_mask = ~lagged_traffic.isna() & ~valid_data[pollutant].isna()
                
                if valid_mask.sum() > 10:  # Ensure enough data points
                    corr, p = pearsonr(lagged_traffic[valid_mask], valid_data[pollutant][valid_mask])
                    correlations.append((lag, corr, p))
            
            # Create DataFrame with lag results
            lag_df = pd.DataFrame(correlations, columns=['lag', 'correlation', 'p_value'])
            
            # Find the lag with the strongest correlation
            max_corr_row = lag_df.iloc[lag_df['correlation'].abs().idxmax()]
            best_lag = max_corr_row['lag']
            best_corr = max_corr_row['correlation']
            best_p = max_corr_row['p_value']
            
            print(f"Best lag for {pollutant}: {best_lag} hours (r = {best_corr:.4f}, p = {best_p:.4f})")
            
            # Plot lagged correlations
            plt.figure(figsize=(12, 6))
            bars = plt.bar(lag_df['lag'], lag_df['correlation'], 
                          color=['green' if c > 0 else 'red' for c in lag_df['correlation']])
            
            # Highlight best lag
            best_idx = lag_df[lag_df['lag'] == best_lag].index[0]
            bars[best_idx].set_color('blue')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f'Lagged Correlation between Traffic Count and {pollutant}\nBest lag: {best_lag} hours (r = {best_corr:.4f})')
            plt.xlabel('Lag (hours)')
            plt.ylabel('Correlation')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{date_str}_{pollutant}_lagged_correlation.png"), dpi=300)
            plt.close()
    
    # Return results
    results = {
        'correlations': pollutant_correlations,
        'p_values': pollutant_p_values,
        'top_pollutants': top_pollutants,
        'detailed_pollutants': detailed_analysis_pollutants
    }
    
    return results
