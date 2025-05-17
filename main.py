from config import LOCATIONS, POLLUTANTS, WEATHER_VARIABLES, TRAFFIC_DATA_FILE
from data_fetcher import get_historical_data
from weather_fetcher import fetch_weather_data
from data_processor import process_air_quality_data
from models.temporal_analysis import temporal_pattern_analysis
from models.spatial_analysis import spatial_comparison_analysis
from models.sarima_analysis import sarima_modeling
from models.weather_correlation_and_mlr import correlation_analysis, multiple_linear_regression
from traffic_data_loader import (load_traffic_data, transform_traffic_data, 
                               find_location_near_hcab, get_traffic_for_date_range, aggregate_hourly_traffic)
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_historical_2014_data():
    """
    Analyze historical 2014 traffic data with matching air quality data.
    """
    print("\n" + "="*80)
    print("HISTORICAL ANALYSIS: 2014 TRAFFIC AND AIR QUALITY")
    print("="*80)
    
    # Step 1: Load and process traffic data
    print("\nLoading and processing 2014 traffic data...")
    raw_traffic_df = load_traffic_data(TRAFFIC_DATA_FILE)
    processed_traffic_df = transform_traffic_data(raw_traffic_df)
    
    if processed_traffic_df is not None:
        # Step 2: Aggregate hourly traffic data
        print("\nAggregating hourly traffic data...")
        aggregated_traffic = aggregate_hourly_traffic(processed_traffic_df)
        
        # Step 3: Find location closest to H.C. Andersens Boulevard
        closest_location_name, location_traffic_df = find_location_near_hcab(aggregated_traffic)
        
        print(f"\nUsing traffic data from {closest_location_name}")
        location_coords = (
            location_traffic_df['latitude'].iloc[0],
            location_traffic_df['longitude'].iloc[0]
        )
        
        # Step 4: Create date range for air quality data (from 2014)
        min_date = location_traffic_df['datetime'].min().strftime('%Y-%m-%d')
        max_date = location_traffic_df['datetime'].max().strftime('%Y-%m-%d')
        print(f"Date range: {min_date} to {max_date}")
        
        # Step 5: Define the single location for air quality data
        historical_location = {
            closest_location_name: location_coords
        }
        
        # Step 6: Fetch historical air quality data
        print(f"\nFetching historical air quality data for {closest_location_name} in 2014...")
        # Reusing the get_historical_data function with specific date range
        air_quality_dfs = get_historical_data(historical_location, POLLUTANTS, 
                                           start_date=min_date, end_date=max_date)
        
        # Step 7: Process air quality data to calculate AQI
        print("\nCalculating Air Quality Index values for historical data...")
        processed_aqi_dfs = process_air_quality_data(air_quality_dfs)
        
        # Get the AQI dataframe for our location
        if closest_location_name in processed_aqi_dfs:
            aqi_df = processed_aqi_dfs[closest_location_name]
            
            # Convert datetime to time if needed for merging
            if 'datetime' in location_traffic_df.columns and 'time' in aqi_df.columns:
                location_traffic_df = location_traffic_df.rename(columns={'datetime': 'time'})
            
            # Step 8: Merge traffic and AQI data
            print("\nMerging historical traffic and air quality data...")
            merged_df = pd.merge(
                location_traffic_df,
                aqi_df,
                on='time',
                how='inner'
            )
            
            print(f"Successfully merged historical data: {len(merged_df)} records")
            
            # Display sample of merged data
            print("\nSample of merged 2014 traffic and AQI data:")
            if len(merged_df) > 0:
                print(merged_df[['time', 'traffic_count', 'AQI', 'Dominant_Pollutant']].head())
                
                # Analyze relationship between 2014 traffic and AQI
                if len(merged_df) >= 100:
                    print("\nPerforming correlation analysis for 2014 data...")
                    corr, lag_corrs = correlation_analysis(merged_df, 
                                                        output_dir="figures/traffic_aqi_2014")
                    
                    if corr is not None:
                        print("\nBuilding traffic-AQI regression model for 2014...")
                        # Select only relevant columns for modeling
                        model_cols = ['traffic_count', 'entry_count'] + [col for col in merged_df.columns if col in WEATHER_VARIABLES]
                        model_df = merged_df[['AQI'] + model_cols].dropna()
                        
                        model, rmse, r2, coef_df = multiple_linear_regression(model_df, 
                                                                          output_dir="figures/traffic_aqi_2014")
                        if model is not None:
                            print(f"\n2014 Traffic-AQI MLR Results:")
                            print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")
                            print("\nVariable importance for 2014 data:")
                            print(coef_df)
            else:
                print("No matching records found between 2014 traffic and air quality data")
        else:
            print(f"No air quality data processed for {closest_location_name}")
    
    print("\n2014 Historical analysis complete.")
    
def main():
    """Main function to execute the AirSense Copenhagen workflow."""
    # Set matplotlib to auto-close figures to avoid warnings
    plt.rcParams['figure.max_open_warning'] = 50
    
    # First run the historical analysis with 2014 data
    analyze_historical_2014_data()
    
    print("\n" + "="*80)
    print("CURRENT DATA ANALYSIS")
    print("="*80)
    
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
            # Ensure AQI data is sorted chronologically for consistent handling
            df = df.sort_values(by='time', ascending=True)
            processed_dfs[name] = df  # Save the sorted dataframe
            
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
        
        # Ensure weather data is sorted chronologically
        df_weather = df_weather.sort_values(by='time', ascending=True)
        
        # Check for duplicate timestamps in weather data
        duplicate_count = df_weather.duplicated(subset=['time']).sum()
        if duplicate_count > 0:
            print(f"WARNING: Found {duplicate_count} duplicate timestamps in weather data")
            # Remove duplicates, keeping first occurrence
            df_weather = df_weather.drop_duplicates(subset=['time'], keep='first')
            print(f"After removing duplicates: {len(df_weather)} weather records")
        
        # Display first 5 rows of weather data
        print("\nFirst 5 rows of weather data (chronological order):")
        print(df_weather.head())
        
        # Check for any missing values
        missing_vals = df_weather.isna().sum()
        if any(missing_vals > 0):
            print("\nMissing values in weather data:")
            print(missing_vals[missing_vals > 0])
            
            # Fill missing weather values with interpolation
            print("Filling missing values with interpolation...")
            
            # Set the time column as index for time-based interpolation
            df_weather = df_weather.set_index('time')
            
            # Perform interpolation on each column
            for col in df_weather.columns:
                df_weather[col] = df_weather[col].interpolate(method='time')
            
            # Reset index to get time column back as a regular column
            df_weather = df_weather.reset_index()
            
            # Check if any NaNs remain after interpolation
            missing_after = df_weather.isna().sum()
            if any(missing_after > 0):
                print("Values still missing after interpolation (will use ffill/bfill):")
                print(missing_after[missing_after > 0])
                # Forward and backward fill for any remaining NaNs
                df_weather = df_weather.ffill().bfill()
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
        # Reference the sorted AQI data
        aqi_df = processed_dfs[hcab_location]
        
        # Check for duplicate timestamps in AQI data
        duplicate_count = aqi_df.duplicated(subset=['time']).sum()
        if duplicate_count > 0:
            print(f"WARNING: Found {duplicate_count} duplicate timestamps in AQI data")
            # Remove duplicates, keeping first occurrence
            aqi_df = aqi_df.drop_duplicates(subset=['time'], keep='first')
            print(f"After removing duplicates: {len(aqi_df)} AQI records")
        
        # Show AQI data date range
        print(f"AQI data time range: {aqi_df['time'].min()} to {aqi_df['time'].max()}")
        
        # Verify that timestamps are in the same format
        print(f"AQI timestamp format example: {aqi_df['time'].iloc[0]}")
        print(f"Weather timestamp format example: {df_weather['time'].iloc[0]}")
        
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
        
        # Merge on timestamp - both datasets are now in chronological order
        merged_df = pd.merge(
            aqi_df,
            df_weather,
            on='time',
            how='inner'
        )
        
        # Check merge results
        merge_count = len(merged_df)
        if merge_count > 0:
            # Sort the merged data by time for consistent handling
            merged_df = merged_df.sort_values(by='time', ascending=True)
            
            print(f"Successfully merged data with {merge_count} matching timestamps")
            print(f"Merged data shape: {merged_df.shape}")
            
            # Display first 5 rows of merged data
            print("\nFirst 5 rows of merged data (chronological order):")
            weather_cols = [col for col in WEATHER_VARIABLES if col in merged_df.columns]
            if len(weather_cols) > 3:
                display_weather_cols = weather_cols[:3]  # Limit to first 3 weather variables
            else:
                display_weather_cols = weather_cols
            columns_to_show = ['time', 'AQI', 'AQI_Category'] + display_weather_cols
            print(merged_df[columns_to_show].head())
            
            # Check for NaN values in critical columns
            nan_check = merged_df[['AQI'] + weather_cols].isna().sum()
            if any(nan_check > 0):
                print("\nWarning: Merged data contains NaN values:")
                print(nan_check[nan_check > 0])
                
                merged_dfs[hcab_location] = merged_df
            else:
                print("\nMerged data has no NaN values in critical columns")
                merged_dfs[hcab_location] = merged_df
        else:
            print("ERROR: Merge resulted in 0 matching rows - no overlapping timestamps!")
            print("Check that your AQI and weather data cover the same time period.")
            return None
    
    # # Other analysis modules - currently commented out
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
    
    print("\nCurrent data analysis complete. Results saved to figures/ directory.")
    
    return merged_dfs

if __name__ == "__main__":
    merged_dfs = main()
