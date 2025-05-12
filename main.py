from config import LOCATIONS, POLLUTANTS
from data_fetcher import get_historical_data
from data_processor import process_air_quality_data
from models import temporal_pattern_analysis, spatial_comparison_analysis, sarima_modeling

def main():
    """Main function to execute the AirSense Copenhagen workflow."""
    # Fetch historical data
    print("Starting AirSense Copenhagen data collection")
    location_dfs = get_historical_data(LOCATIONS, POLLUTANTS)
    
    # Process data and calculate AQI
    print("\nCalculating Air Quality Index values")
    processed_dfs = process_air_quality_data(location_dfs)
    
    # Display first 5 rows for each dataframe with AQI
    for name, df in processed_dfs.items():
        print(f"\nFirst 5 rows for {name}:")
        display_columns = ['time', 'AQI', 'AQI_Category', 'Dominant_Pollutant']
        print(df[display_columns].head())
    
    # Execute modeling and analysis tasks
    print("\nPerforming temporal pattern analysis...")
    temporal_pattern_analysis(processed_dfs)
    
    print("\nPerforming spatial comparison analysis...")
    spatial_comparison_analysis(processed_dfs)
    
    print("\nPerforming SARIMA modeling and forecasting...")
    sarima_modeling(processed_dfs)
    
    print("\nAnalysis complete. Results saved to figures/ directory.")
    
    return processed_dfs

if __name__ == "__main__":
    processed_dfs = main()
