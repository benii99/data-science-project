import pandas as pd
from config import MIN_HOURS_24H, MIN_HOURS_8H
from aqi_calculator import pollutant_aqi, get_aqi_category

def calculate_rolling_averages(df):
    """
    Calculate rolling averages as per EPA standards.
    
    Parameters:
    df: DataFrame with hourly pollutant data
    
    Returns:
    DataFrame with added columns for averaged values
    """
    # Sort by time ascending for proper rolling calculations
    df = df.sort_values(by="time", ascending=True)
    
    # 24-hour averages for PM2.5 and PM10
    df['pm2_5_24h'] = df['pm2_5'].rolling(window=24, min_periods=MIN_HOURS_24H).mean()
    df['pm10_24h'] = df['pm10'].rolling(window=24, min_periods=MIN_HOURS_24H).mean()
    
    # 8-hour averages for O3 and CO
    df['ozone_8h'] = df['ozone'].rolling(window=8, min_periods=MIN_HOURS_8H).mean()
    df['carbon_monoxide_8h'] = df['carbon_monoxide'].rolling(window=8, min_periods=MIN_HOURS_8H).mean()
    
    # 1-hour values for NO2 and SO2 (already hourly)
    df['nitrogen_dioxide_1h'] = df['nitrogen_dioxide']
    df['sulphur_dioxide_1h'] = df['sulphur_dioxide']
    
    # Sort back to descending order for display
    df = df.sort_values(by="time", ascending=False)
    return df

def calculate_aqi(df):
    """
    Calculate AQI and dominant pollutant for each row.
    
    Parameters:
    df: DataFrame with properly averaged pollutant values
    
    Returns:
    DataFrame with added AQI columns
    """
    aqi_values = []
    dominant_pollutants = []
    
    for idx, row in df.iterrows():
        pollutant_aqis = {}
        
        # Use the averaged values for AQI calculation
        if not pd.isna(row['pm2_5_24h']):
            aqi = pollutant_aqi('pm2_5', row['pm2_5_24h'])
            if aqi is not None:
                pollutant_aqis['PM2.5'] = aqi
                
        if not pd.isna(row['pm10_24h']):
            aqi = pollutant_aqi('pm10', row['pm10_24h'])
            if aqi is not None:
                pollutant_aqis['PM10'] = aqi
                
        if not pd.isna(row['nitrogen_dioxide_1h']):
            aqi = pollutant_aqi('nitrogen_dioxide', row['nitrogen_dioxide_1h'])
            if aqi is not None:
                pollutant_aqis['NO2'] = aqi
                
        if not pd.isna(row['carbon_monoxide_8h']):
            aqi = pollutant_aqi('carbon_monoxide', row['carbon_monoxide_8h'])
            if aqi is not None:
                pollutant_aqis['CO'] = aqi
                
        if not pd.isna(row['sulphur_dioxide_1h']):
            aqi = pollutant_aqi('sulphur_dioxide', row['sulphur_dioxide_1h'])
            if aqi is not None:
                pollutant_aqis['SO2'] = aqi
                
        if not pd.isna(row['ozone_8h']):
            aqi = pollutant_aqi('ozone', row['ozone_8h'])
            if aqi is not None:
                pollutant_aqis['O3'] = aqi
        
        if pollutant_aqis:
            max_aqi = max(pollutant_aqis.values())
            dominant_pollutant = [p for p, v in pollutant_aqis.items() if v == max_aqi][0]
            aqi_values.append(max_aqi)
            dominant_pollutants.append(dominant_pollutant)
        else:
            aqi_values.append(None)
            dominant_pollutants.append(None)
    
    df['AQI'] = aqi_values
    df['Dominant_Pollutant'] = dominant_pollutants
    df['AQI_Category'] = df['AQI'].apply(get_aqi_category)
    
    return df

def process_air_quality_data(location_dfs):
    """
    Process air quality data for all locations.
    
    Parameters:
    location_dfs: Dictionary of location names and DataFrames
    
    Returns:
    Dictionary of location names and processed DataFrames
    """
    processed_dfs = {}
    
    for name, df in location_dfs.items():
        print(f"Processing data for {name}...")
        # Calculate rolling averages
        processed_df = calculate_rolling_averages(df)
        
        # Calculate AQI and add categories
        processed_df = calculate_aqi(processed_df)
        
        processed_dfs[name] = processed_df
        print(f"Completed processing for {name}")
    
    return processed_dfs
