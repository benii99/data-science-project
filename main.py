from config import LOCATIONS, POLLUTANTS, WEATHER_VARIABLES
from data_fetcher import get_historical_data
from weather_fetcher import fetch_weather_data
from data_processor import process_air_quality_data
from models.temporal_analysis import temporal_pattern_analysis
from models.spatial_analysis import spatial_comparison_analysis
from models.sarima_analysis import sarima_modeling
from models.weather_correlation_and_mlr import correlation_analysis, multiple_linear_regression
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    """Main function to execute the AirSense Copenhagen workflow."""
    # Set matplotlib to auto-close figures to avoid warnings
    plt.rcParams['figure.max_open_warning'] = 50
    
    # Fetch historical air quality data for all locations
    print("Starting AirSense Copenhagen data collection")
    location_dfs = get_historical_data(LOCATIONS, POLLUTANTS)
    
    # Process data and calculate AQI for all locations
    print("\nCalculating Air Quality Index values")
    processed_dfs = process_air_quality_data(location_dfs)
    
    # Display first 5 rows for each dataframe with AQI and determine date range
    hcab_location = "H.C. Andersens Boulevard"
    aqi_start_date = None
    aqi_end_date = None
    
    for name, df in processed_dfs.items():
        print(f"\nFirst 5 rows of processed AQI data for {name}:")
        display_columns = ['time', 'AQI', 'AQI_Category', 'Dominant_Pollutant']
        print(df[display_columns].head())
        
        # Get the date range from the data
        if name == hcab_location and len(df) > 0:
            aqi_start_date = df['time'].min().strftime('%Y-%m-%d')
            aqi_end_date = df['time'].max().strftime('%Y-%m-%d')
            print(f"AQI data spans: {aqi_start_date} to {aqi_end_date}")
    
    # Fetch historical weather data ONLY for H.C. Andersens Boulevard
    print(f"\nFetching weather data for {hcab_location} only")
    latitude, longitude = LOCATIONS[hcab_location]
    
    # Use the EXACT same date range as AQI data
    print(f"Using the exact same date range as AQI data: {aqi_start_date} to {aqi_end_date}")
    df_weather = fetch_weather_data(latitude, longitude, 
                                   start_date=aqi_start_date, 
                                   end_date=aqi_end_date)
    
    if df_weather is not None:
        print(f"Retrieved {len(df_weather)} weather records for {hcab_location}")
        
        # Check for duplicate timestamps in weather data
        duplicate_count = df_weather.duplicated(subset=['time']).sum()
        if duplicate_count > 0:
            print(f"WARNING: Found {duplicate_count} duplicate timestamps in weather data")
            # Remove duplicates, keeping first occurrence
            df_weather = df_weather.drop_duplicates(subset=['time'], keep='first')
            print(f"After removing duplicates: {len(df_weather)} weather records")
        
        # Display first 5 rows of weather data
        print("\nFirst 5 rows of weather data:")
        print(df_weather.head())
        
        # Check for any missing values
        missing_vals = df_weather.isna().sum()
        if any(missing_vals > 0):
            print("\nMissing values in weather data:")
            print(missing_vals[missing_vals > 0])
        else:
            print("\nNo missing values in weather data")
        
        # Show weather data date range
        print(f"Weather data time range: {df_weather['time'].min()} to {df_weather['time'].max()}")
    else:
        print(f"Failed to retrieve weather data for {hcab_location}")
        return None
    
    # Merge weather and AQI data by timestamp ONLY for H.C. Andersens Boulevard
    print(f"\nMerging weather and air quality datasets for {hcab_location}")
    merged_dfs = {}
    
    if hcab_location in processed_dfs:
        # Check for duplicate timestamps in AQI data
        aqi_df = processed_dfs[hcab_location]
        duplicate_count = aqi_df.duplicated(subset=['time']).sum()
        if duplicate_count > 0:
            print(f"WARNING: Found {duplicate_count} duplicate timestamps in AQI data")
            # Remove duplicates, keeping first occurrence
            aqi_df = aqi_df.drop_duplicates(subset=['time'], keep='first')
            print(f"After removing duplicates: {len(aqi_df)} AQI records")
        
        # Show AQI data date range
        print(f"AQI data time range: {aqi_df['time'].min()} to {aqi_df['time'].max()}")
        
        # Check for overlap in date ranges
        aqi_start = aqi_df['time'].min()
        aqi_end = aqi_df['time'].max()
        weather_start = df_weather['time'].min()
        weather_end = df_weather['time'].max()
        
        if aqi_end < weather_start or weather_end < aqi_start:
            print("WARNING: No overlap between AQI and weather data time ranges!")
            print(f"AQI: {aqi_start} to {aqi_end}")
            print(f"Weather: {weather_start} to {weather_end}")
        else:
            overlap_start = max(aqi_start, weather_start)
            overlap_end = min(aqi_end, weather_end)
            print(f"Data overlap period: {overlap_start} to {overlap_end}")
        
        # Merge on timestamp
        merged_df = pd.merge(
            aqi_df,
            df_weather,
            on='time',
            how='inner'
        )
        
        # Check merge results
        merge_count = len(merged_df)
        if merge_count > 0:
            print(f"Successfully merged data with {merge_count} matching timestamps")
            print(f"Merged data shape: {merged_df.shape}")
            
            # Display first 5 rows of merged data
            print("\nFirst 5 rows of merged data (AQI + weather):")
            weather_cols = [col for col in WEATHER_VARIABLES if col in merged_df.columns]
            if len(weather_cols) > 3:
                display_weather_cols = weather_cols[:3]  # Limit to first 3 weather variables
            else:
                display_weather_cols = weather_cols
            columns_to_show = ['time', 'AQI', 'AQI_Category'] + display_weather_cols
            print(merged_df[columns_to_show].head())
            
            # Check for NaN values in critical columns and handle them
            nan_check = merged_df[['AQI'] + weather_cols].isna().sum()
            if any(nan_check > 0):
                print("\nWarning: Merged data contains NaN values:")
                print(nan_check[nan_check > 0])
                
                # Impute missing weather values
                for col in weather_cols:
                    if merged_df[col].isna().sum() > 0:
                        # Use forward and backward fill to handle missing values
                        merged_df[col] = merged_df[col].ffill().bfill()
                
                # Check if any NaNs remain after imputation
                nan_check_after = merged_df[['AQI'] + weather_cols].isna().sum()
                if any(nan_check_after > 0):
                    print("NaN values after imputation:")
                    print(nan_check_after[nan_check_after > 0])
                    # If NaNs remain, remove rows with NaNs in critical columns
                    merged_df = merged_df.dropna(subset=['AQI'] + weather_cols)
                    print(f"After removing rows with NaNs: {len(merged_df)} records")
            else:
                print("\nMerged data has no NaN values in critical columns")
            
            merged_dfs[hcab_location] = merged_df
        else:
            print("ERROR: Merge resulted in 0 matching rows - no overlapping timestamps!")
            print("Check that your AQI and weather data cover the same time period.")
            return None
    
    # Other analysis modules - currently commented out
    # print("\nPerforming temporal pattern analysis...")
    # temporal_pattern_analysis(processed_dfs)
    
    # print("\nPerforming spatial comparison analysis...")
    # spatial_comparison_analysis(processed_dfs)
    
    # print("\nPerforming SARIMA modeling and forecasting...")
    # sarima_modeling(processed_dfs)
    
    # Weather and AQI correlation analysis and MLR modeling
    if hcab_location in merged_dfs and len(merged_dfs[hcab_location]) >= 100:
        print("\nPerforming weather-AQI correlation analysis...")
        corr, lag_corrs = correlation_analysis(merged_dfs[hcab_location], 
                                             output_dir="figures/weather_aqi")
        
        if corr is not None:
            print("\nBuilding multiple linear regression model...")
            model, rmse, r2, coef_df = multiple_linear_regression(merged_dfs[hcab_location], 
                                                               output_dir="figures/weather_aqi")
            if model is not None:
                print(f"\nWeather-AQI MLR Results:")
                print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")
                print("\nWeather variable importance:")
                if coef_df is not None:
                    print(coef_df)
                else:
                    print("No coefficient data available")
            else:
                print("Failed to build MLR model")
        else:
            print("Failed to perform correlation analysis")
    elif hcab_location in merged_dfs:
        print(f"\nWARNING: Only {len(merged_dfs[hcab_location])} data points available.")
        print("Insufficient data for reliable statistical analysis.")
        if len(merged_dfs[hcab_location]) > 0:
            print("Will proceed with limited data, but results may not be statistically valid.")
            
            print("\nPerforming weather-AQI correlation analysis with limited data...")
            corr, lag_corrs = correlation_analysis(merged_dfs[hcab_location], 
                                                output_dir="figures/weather_aqi")
            
            print("\nBuilding multiple linear regression model with limited data...")
            model, rmse, r2, coef_df = multiple_linear_regression(merged_dfs[hcab_location], 
                                                              output_dir="figures/weather_aqi")
            if model is not None:
                print(f"\nWeather-AQI MLR Results (CAUTION - limited data):")
                print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")
                print("\nWeather variable importance:")
                print(coef_df)
    else:
        print("No valid merged data available for weather-AQI analysis")
    
    print("\nAnalysis complete. Results saved to figures/ directory.")
    
    return merged_dfs

if __name__ == "__main__":
    merged_dfs = main()
