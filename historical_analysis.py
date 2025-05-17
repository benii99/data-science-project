import pandas as pd
from datetime import datetime
from config import POLLUTANTS
from data_fetcher import get_historical_data
from data_processor import process_air_quality_data
from traffic_data_loader import load_traffic_data, transform_traffic_data, aggregate_hourly_traffic, find_location_near_hcab

def fetch_historical_aqi_for_traffic(traffic_file, year=2014):
    """
    Fetch historical air quality data to match with traffic data from 2014.
    
    Parameters:
    traffic_file: Path to the Excel file with traffic data
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
        
        # Step 3: Find location closest to H.C. Andersens Boulevard
        closest_location_name, location_traffic_df = find_location_near_hcab(aggregated_traffic)
        
        print(f"\nUsing traffic data from {closest_location_name}")
        location_coords = (
            location_traffic_df['latitude'].iloc[0],
            location_traffic_df['longitude'].iloc[0]
        )
        
        # Step 4: Create date range for air quality data
        min_date = location_traffic_df['datetime'].min().strftime('%Y-%m-%d')
        max_date = location_traffic_df['datetime'].max().strftime('%Y-%m-%d')
        print(f"Date range: {min_date} to {max_date}")
        
        # Step 5: Define the single location for air quality data
        historical_location = {
            closest_location_name: location_coords
        }
        
        # Step 6: Fetch historical air quality data
        print(f"\nFetching historical air quality data for {closest_location_name} in {year}...")
        # Reusing the get_historical_data function but with specific date range
        air_quality_dfs = get_historical_data(historical_location, POLLUTANTS, 
                                            start_date=min_date, end_date=max_date)
        
        # Step 7: Process air quality data to calculate AQI
        print("\nCalculating Air Quality Index values for historical data...")
        processed_aqi_dfs = process_air_quality_data(air_quality_dfs)
        
        # Get the AQI dataframe for our location
        aqi_df = processed_aqi_dfs[closest_location_name]
        
        # Step 8: Merge traffic and AQI data
        print("\nMerging traffic and air quality data...")
        merged_df = pd.merge(
            location_traffic_df,
            aqi_df,
            on='time' if 'time' in aqi_df.columns else 'datetime',
            how='inner'
        )
        
        print(f"Successfully merged data: {len(merged_df)} records")
        
        return location_traffic_df, aqi_df, merged_df
    
    return None, None, None
