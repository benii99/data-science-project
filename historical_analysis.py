import pandas as pd
from datetime import datetime
from config import POLLUTANTS
from data_fetcher import get_historical_data
from data_processor import process_air_quality_data
from traffic_data_loader import load_traffic_data, transform_traffic_data, aggregate_hourly_traffic, filter_traffic_by_location

def fetch_historical_aqi_for_traffic(traffic_file, location_name="Torvegade", year=2014):
    """
    Fetch historical air quality data to match with traffic data from 2014.
    
    Parameters:
    traffic_file: Path to the Excel file with traffic data
    location_name: Name of location to use (default: Torvegade)
    year: Year of the traffic data
    
    Returns:
    tuple of (traffic_df, aqi_df, merged_df)
    """
    # Step 1: Load and process traffic data
    print(f"\nLoading and processing traffic data for {year}...")
    raw_traffic_df = load_traffic_data(traffic_file)
    processed_traffic_df = transform_traffic_data(raw_traffic_df)
    
    if processed_traffic_df is not None:
        # Step 2: Aggregate hourly traffic data
        print("\nAggregating hourly traffic data...")
        aggregated_traffic = aggregate_hourly_traffic(processed_traffic_df)
        
        # Step 3: Filter for Torvegade location data
        print(f"\nFiltering for {location_name} traffic data...")
        location_traffic_df = filter_traffic_by_location(aggregated_traffic, location_name)
        
        # Check if we found data for the location
        if location_traffic_df is None or len(location_traffic_df) == 0:
            print(f"Error: No traffic data found for {location_name}")
            return None, None, None
        
        print(f"\nUsing traffic data from {location_name}")
        location_coords = (
            location_traffic_df['latitude'].iloc[0],
            location_traffic_df['longitude'].iloc[0]
        )
        
        # Step 4: Create date range for air quality data
        if 'datetime' in location_traffic_df.columns:
            min_date = location_traffic_df['datetime'].min().strftime('%Y-%m-%d')
            max_date = location_traffic_df['datetime'].max().strftime('%Y-%m-%d')
        else:
            min_date = location_traffic_df['time'].min().strftime('%Y-%m-%d')
            max_date = location_traffic_df['time'].max().strftime('%Y-%m-%d')
            
        print(f"Date range: {min_date} to {max_date}")
        
        # Step 5: Define the single location for air quality data
        historical_location = {
            location_name: location_coords
        }
        
        # Step 6: Fetch historical air quality data
        print(f"\nFetching historical air quality data for {location_name} in {year}...")
        # Reusing the get_historical_data function but with specific date range
        air_quality_dfs = get_historical_data(historical_location, POLLUTANTS, 
                                            start_date=min_date, end_date=max_date)
        
        # Step 7: Process air quality data to calculate AQI
        print("\nCalculating Air Quality Index values for historical data...")
        processed_aqi_dfs = process_air_quality_data(air_quality_dfs)
        
        # Get the AQI dataframe for our location
        if location_name not in processed_aqi_dfs:
            print(f"Error: No AQI data processed for {location_name}")
            return location_traffic_df, None, None
            
        aqi_df = processed_aqi_dfs[location_name]
        
        # Step 8: Merge traffic and AQI data
        print("\nMerging traffic and air quality data...")
        # Ensure the time column names match before merging
        if 'datetime' in location_traffic_df.columns and 'time' in aqi_df.columns:
            location_traffic_df = location_traffic_df.rename(columns={'datetime': 'time'})
            
        merged_df = pd.merge(
            location_traffic_df,
            aqi_df,
            on='time',
            how='inner'
        )
        
        print(f"Successfully merged data: {len(merged_df)} records")
        
        return location_traffic_df, aqi_df, merged_df
    
    return None, None, None
