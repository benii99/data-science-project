import requests
import pandas as pd
from datetime import datetime, timedelta
from config import AIR_QUALITY_API_URL, DEFAULT_HISTORY_DAYS

def fetch_air_quality_data(latitude, longitude, pollutants, start_date, end_date):
    """
    Fetch air quality data from Open-Meteo API.
    
    Parameters:
    latitude, longitude: Location coordinates
    pollutants: Comma-separated pollutant names
    start_date, end_date: Date range in YYYY-MM-DD format
    
    Returns:
    DataFrame with hourly air quality data
    """
    url = f"{AIR_QUALITY_API_URL}?latitude={latitude}&longitude={longitude}&hourly={pollutants}&start_date={start_date}&end_date={end_date}&timezone=auto"
    
    response = requests.get(url)
    if not response.ok:
        print(f"Error fetching data: {response.status_code}")
        return None
        
    data = response.json()
    hourly_data = data.get("hourly", {})
    df = pd.DataFrame(hourly_data)
    
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    
    return df

def get_historical_data(locations, pollutants, days=None):
    """
    Fetch historical air quality data for multiple locations.
    
    Parameters:
    locations: Dictionary of location names and coordinates
    pollutants: Comma-separated pollutant names
    days: Number of days of historical data to retrieve (defaults to config value)
    
    Returns:
    Dictionary of location names and corresponding DataFrames
    """
    if days is None:
        days = DEFAULT_HISTORY_DAYS
        
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    location_dfs = {}
    
    for name, (latitude, longitude) in locations.items():
        print(f"Fetching data for {name}...")
        df = fetch_air_quality_data(latitude, longitude, pollutants, start_date, end_date)
        if df is not None:
            location_dfs[name] = df
            print(f"Retrieved {len(df)} records for {name}")
        else:
            print(f"Failed to retrieve data for {name}")
    
    return location_dfs
