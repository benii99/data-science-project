import requests
import pandas as pd
from datetime import datetime, timedelta
from config import WEATHER_API_URL, WEATHER_VARIABLES, DEFAULT_HISTORY_DAYS

def fetch_weather_data(latitude, longitude, start_date=None, end_date=None, timezone="auto", adjust_year=None):
    """
    Fetch historical weather data from Open-Meteo API.
    
    Parameters:
    latitude, longitude: Location coordinates
    start_date, end_date: Date range in YYYY-MM-DD format (if None, uses DEFAULT_HISTORY_DAYS)
    timezone: Timezone for the data (default: "auto")
    adjust_year: If provided, adjusts dates to the specified year (for aligning datasets)
    
    Returns:
    DataFrame with hourly weather data
    """
    # If dates are not provided, calculate them based on DEFAULT_HISTORY_DAYS
    if start_date is None or end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=DEFAULT_HISTORY_DAYS)).strftime("%Y-%m-%d")
    
    print(f"Requesting weather data from {start_date} to {end_date}")
    
    # Construct API URL
    url = f"{WEATHER_API_URL}?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly={','.join(WEATHER_VARIABLES)}&timezone={timezone}"
    
    # Make the request
    response = requests.get(url)
    if not response.ok:
        print(f"Error fetching weather data: {response.status_code}")
        return None
    
    data = response.json()
    
    # Convert to DataFrame
    hourly_data = data.get("hourly", {})
    df = pd.DataFrame(hourly_data)
    
    if not df.empty and 'time' in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        
        # Adjust year if specified (to align with other datasets)
        if adjust_year is not None:
            df["time"] = df["time"].apply(lambda x: x.replace(year=adjust_year))
            print(f"Adjusted weather data years to {adjust_year}")
    
    return df

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
