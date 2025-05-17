import pandas as pd
import numpy as np
from datetime import datetime
import os
from math import radians, cos, sin, asin, sqrt

def load_traffic_data(file_path):
    """
    Load traffic data from the Excel file and perform initial cleaning.
    
    Parameters:
    file_path: Path to the Excel file with traffic data
    
    Returns:
    DataFrame with cleaned traffic data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None
    
    # Read the Excel file
    print(f"Loading traffic data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # Print the initial shape and columns for verification
    print(f"Raw traffic data shape: {df.shape}")
    print(f"Traffic data columns: {df.columns.tolist()[:10]}...")
    
    return df

def transform_traffic_data(df):
    """
    Transform and clean the traffic data into a tidy format.
    
    Parameters:
    df: Raw DataFrame from the Excel file
    
    Returns:
    DataFrame in tidy format with proper data types
    """
    if df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Standardize column names if needed
    column_mapping = {}
    for col in df_clean.columns:
        if 'road' in col.lower() or 'name' in col.lower():
            column_mapping[col] = 'road_name'
        elif 'xcoord' in col.lower() or 'x_coord' in col.lower() or 'x-coord' in col.lower():
            column_mapping[col] = 'xcoord'
        elif 'ycoord' in col.lower() or 'y_coord' in col.lower() or 'y-coord' in col.lower():
            column_mapping[col] = 'ycoord'
        elif 'date' in col.lower():
            column_mapping[col] = 'date'
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    # Ensure we have the expected columns
    essential_columns = ['road_name', 'xcoord', 'ycoord', 'date']
    for col in essential_columns:
        if col not in df_clean.columns:
            print(f"Error: Required column '{col}' not found in dataset")
            return None
    
    # Convert date to datetime format
    try:
        df_clean['date'] = pd.to_datetime(df_clean['date'], format='%d.%m.%Y')
    except:
        try:
            # Try alternative parsing
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            print("Date format was different than expected, but conversion succeeded.")
        except:
            print("Warning: Could not parse date column. Please check the format.")
    
    # Convert coordinates from Danish projection to standard lat/lon
    df_clean = convert_coordinates(df_clean)
    
    # Identify hour columns (those with format XX:XX-XX:XX)
    hour_columns = [col for col in df_clean.columns if '-' in str(col) and ':' in str(col)]
    
    if not hour_columns:
        print("Warning: No hour columns detected. Looking for numbered columns...")
        # Try to find columns that might represent hours (0-23)
        hour_columns = [col for col in df_clean.columns if str(col).isdigit() and int(col) >= 0 and int(col) <= 23]
    
    if not hour_columns:
        print("Error: Could not identify hourly traffic data columns")
        return None
    
    # Reshape the dataframe to have one row per hour (long format)
    df_tidy = pd.melt(
        df_clean, 
        id_vars=['road_name', 'xcoord', 'ycoord', 'date', 'latitude', 'longitude'], 
        value_vars=hour_columns,
        var_name='hour_range', 
        value_name='traffic_count'
    )
    
    # Extract the hour from hour_range
    def extract_hour(hour_str):
        """Extract the starting hour from formats like '00:00-01:00' or '0'"""
        if ':' in str(hour_str) and '-' in str(hour_str):
            return int(hour_str.split(':')[0])
        elif str(hour_str).isdigit():
            return int(hour_str)
        else:
            try:
                # Try to convert directly to integer
                return int(float(hour_str))
            except:
                return np.nan
    
    df_tidy['hour'] = df_tidy['hour_range'].apply(extract_hour)
    
    # Create a full datetime column
    df_tidy['datetime'] = pd.to_datetime(df_tidy['date'].astype(str) + ' ' + 
                                      df_tidy['hour'].astype(str) + ':00:00', errors='coerce')
    
    # Drop rows where datetime creation failed
    invalid_rows = df_tidy['datetime'].isna().sum()
    if invalid_rows > 0:
        print(f"Warning: Dropping {invalid_rows} rows with invalid datetime values")
        df_tidy = df_tidy.dropna(subset=['datetime'])
    
    # Sort by location and time
    df_tidy = df_tidy.sort_values(['road_name', 'datetime'])
    
    # Check for missing values
    na_count = df_tidy['traffic_count'].isna().sum()
    if na_count > 0:
        print(f"Warning: Found {na_count} missing traffic count values")
    
    print(f"Transformed traffic data shape: {df_tidy.shape}")
    print(f"Date range: {df_tidy['date'].min()} to {df_tidy['date'].max()}")
    print(f"Number of unique locations: {df_tidy['road_name'].nunique()}")
    
    return df_tidy

def aggregate_hourly_traffic(df_tidy):
    """
    Aggregate traffic data by datetime and location using average times three approach.
    This accounts for hours with fewer than three entries while normalizing the data.
    
    Parameters:
    df_tidy: Cleaned and transformed traffic DataFrame
    
    Returns:
    DataFrame with aggregated hourly traffic data
    """
    # Group by datetime and road_name
    grouped = df_tidy.groupby(['datetime', 'road_name'])
    
    # Calculate mean traffic count and multiply by 3
    aggregated = grouped.agg({
        'traffic_count': 'mean',
        'latitude': 'first',
        'longitude': 'first',
        'xcoord': 'first',
        'ycoord': 'first',
        'date': 'first'
    }).reset_index()
    
    # Multiply mean by 3 to normalize the data
    aggregated['traffic_count'] = aggregated['traffic_count'] * 3
    
    # Count entries per hour for verification
    entries_per_hour = grouped.size().reset_index(name='entry_count')
    
    # Merge to get count of entries
    aggregated = pd.merge(
        aggregated, 
        entries_per_hour, 
        on=['datetime', 'road_name']
    )
    
    print(f"Aggregated traffic data shape: {aggregated.shape}")
    print(f"Average number of entries per hour: {aggregated['entry_count'].mean():.2f}")
    
    return aggregated

def convert_coordinates(df):
    """
    Convert coordinates from Danish projection to WGS84 (latitude/longitude).
    The input coordinates are likely in ETRS89 / UTM zone 32N (EPSG:25832).
    
    Parameters:
    df: DataFrame with xcoord, ycoord columns
    
    Returns:
    DataFrame with added latitude and longitude columns
    """
    # Add empty columns for lat/lon
    df['latitude'] = np.nan
    df['longitude'] = np.nan
    
    try:
        # Try to use pyproj if available
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        
        # Convert each coordinate pair
        for idx, row in df.iterrows():
            try:
                lon, lat = transformer.transform(row['xcoord'], row['ycoord'])
                df.at[idx, 'longitude'] = lon
                df.at[idx, 'latitude'] = lat
            except Exception as e:
                print(f"Error converting coordinates for row {idx}: {e}")
    except ImportError:
        print("pyproj not available, using approximate conversion...")
        # Simple approximation for Denmark (not precise but functional)
        # These values are specific to UTM zone 32N and only work approximately in Denmark
        for idx, row in df.iterrows():
            try:
                # Very rough approximation for Denmark
                lon = (row['xcoord'] - 500000) / 65000 + 9.5
                lat = row['ycoord'] / 111000 + 51.3
                df.at[idx, 'longitude'] = lon
                df.at[idx, 'latitude'] = lat
            except Exception as e:
                print(f"Error approximating coordinates for row {idx}: {e}")
    
    return df

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km

def get_available_locations(df_tidy):
    """
    Get a list of unique locations in the traffic dataset with their coordinates.
    
    Parameters:
    df_tidy: Cleaned traffic DataFrame
    
    Returns:
    Dictionary of location names and their coordinates
    """
    locations = {}
    
    # Group by road name and get coordinates
    for name, group in df_tidy.groupby('road_name'):
        # Get the first row's coordinates (they should be the same for the same road)
        lat = group['latitude'].iloc[0]
        lon = group['longitude'].iloc[0]
        locations[name] = (lat, lon)
        
    return locations

def find_nearest_location(df_tidy, target_lat, target_lon):
    """
    Find the location in the dataset closest to the given coordinates.
    
    Parameters:
    df_tidy: Cleaned traffic DataFrame
    target_lat, target_lon: Target coordinates to match
    
    Returns:
    Name of the closest location and its distance in km
    """
    locations = get_available_locations(df_tidy)
    
    distances = {}
    for name, (lat, lon) in locations.items():
        distance = haversine(target_lat, lon1=target_lon, lat2=lat, lon2=lon)
        distances[name] = distance
    
    # Find the minimum distance
    closest = min(distances, key=distances.get)
    return closest, distances[closest]

def find_location_near_hcab(df_tidy, hcab_lat=55.6761, hcab_lon=12.5683):
    """
    Find the traffic measurement location closest to H.C. Andersens Boulevard.
    
    Parameters:
    df_tidy: Cleaned traffic DataFrame
    hcab_lat, hcab_lon: Coordinates of H.C. Andersens Boulevard
    
    Returns:
    Name of the closest location and DataFrame filtered for that location
    """
    closest_name, distance = find_nearest_location(df_tidy, hcab_lat, hcab_lon)
    print(f"Closest location to H.C. Andersens Boulevard is '{closest_name}' at {distance:.2f} km distance")
    
    return closest_name, filter_traffic_by_location(df_tidy, closest_name)

def filter_traffic_by_location(df_tidy, location_name):
    """
    Filter the traffic data for a specific location.
    
    Parameters:
    df_tidy: Cleaned traffic DataFrame
    location_name: Name of the location to filter
    
    Returns:
    DataFrame with traffic data for the specified location
    """
    filtered_df = df_tidy[df_tidy['road_name'] == location_name]
    print(f"Filtered to {len(filtered_df)} records for location '{location_name}'")
    return filtered_df

def get_traffic_for_date_range(df_tidy, location_name, start_date, end_date):
    """
    Get traffic data for a specific location and date range.
    
    Parameters:
    df_tidy: Cleaned traffic DataFrame
    location_name: Name of the location to filter
    start_date, end_date: Date range in 'YYYY-MM-DD' format
    
    Returns:
    DataFrame with traffic data for the specified location and date range
    """
    # Convert string dates to datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Filter by location and date range
    location_data = df_tidy[df_tidy['road_name'] == location_name]
    date_filtered = location_data[
        (location_data['datetime'] >= start_date) & 
        (location_data['datetime'] <= end_date)
    ]
    
    print(f"Retrieved {len(date_filtered)} records for '{location_name}' from {start_date.date()} to {end_date.date()}")
    return date_filtered
