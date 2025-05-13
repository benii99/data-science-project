import requests
import pandas as pd
from datetime import datetime, timedelta
from config import WEATHER_API_URL, WEATHER_VARIABLES, DEFAULT_HISTORY_DAYS

def fetch_weather_data(latitude, longitude, start_date=None, end_date=None, timezone="auto"):
    """
    Fetch historical weather data from Open-Meteo API.
    
    Parameters:
    latitude, longitude: Location coordinates
    start_date, end_date: Date range in YYYY-MM-DD format (if None, uses DEFAULT_HISTORY_DAYS)
    timezone: Timezone for the data (default: "auto")
    
    Returns:
    DataFrame with hourly weather data
    """
    # If dates are not provided, calculate them based on DEFAULT_HISTORY_DAYS
    if start_date is None or end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=DEFAULT_HISTORY_DAYS)).strftime("%Y-%m-%d")
    
    print(f"Requesting weather data from {start_date} to {end_date}")
    
    # API might have limitations on date range, so we'll break up the request by months
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Create empty dataframe to store all results
    all_weather_data = pd.DataFrame()
    
    # Process in 3-month chunks (typical API limitation)
    current_start = start_dt
    while current_start < end_dt:
        # Calculate chunk end (3 months later or end_dt, whichever comes first)
        current_end = min(current_start + timedelta(days=90), end_dt)
        
        chunk_start = current_start.strftime("%Y-%m-%d")
        chunk_end = current_end.strftime("%Y-%m-%d")
        print(f"  Fetching chunk: {chunk_start} to {chunk_end}")
        
        # Construct API URL for this chunk
        url = f"{WEATHER_API_URL}?latitude={latitude}&longitude={longitude}&start_date={chunk_start}&end_date={chunk_end}&hourly={','.join(WEATHER_VARIABLES)}&timezone={timezone}"
        
        # Make the request
        response = requests.get(url)
        if not response.ok:
            print(f"  Error fetching weather data for chunk: {response.status_code}")
        else:
            data = response.json()
            hourly_data = data.get("hourly", {})
            chunk_df = pd.DataFrame(hourly_data)
            
            if not chunk_df.empty and 'time' in chunk_df.columns:
                chunk_df["time"] = pd.to_datetime(chunk_df["time"])
                all_weather_data = pd.concat([all_weather_data, chunk_df])
            
        # Move to next chunk
        current_start = current_end + timedelta(days=1)
    
    # Process the combined data
    if all_weather_data.empty:
        print("No weather data retrieved for the specified period")
        return None
    
    # Remove any duplicates that might have been created at chunk boundaries
    all_weather_data = all_weather_data.drop_duplicates(subset=['time'])
    
    # Sort by time to ensure chronological order
    all_weather_data = all_weather_data.sort_values('time')
    
    return all_weather_data

def get_historical_weather(locations, days=None):
    """
    Fetch historical weather data for multiple locations.
    
    Parameters:
    locations: Dictionary of location names and coordinates
    days: Number of days of historical data to retrieve (uses DEFAULT_HISTORY_DAYS if None)
    
    Returns:
    Dictionary of location names and corresponding weather DataFrames
    """
    if days is None:
        days = DEFAULT_HISTORY_DAYS
        
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    weather_dfs = {}
    
    for name, (latitude, longitude) in locations.items():
        print(f"Fetching weather data for {name}...")
        df = fetch_weather_data(latitude, longitude, start_date, end_date)
        if df is not None:
            weather_dfs[name] = df
            print(f"Retrieved {len(df)} weather records for {name}")
        else:
            print(f"Failed to retrieve weather data for {name}")
    
    return weather_dfs
