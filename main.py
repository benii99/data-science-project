# =============================================================================
# 'AirSense Copenhagen': Data Science Project
# Business Intelligence cand. merc. 
# =============================================================================

# Standard library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Project configuration imports
from config import LOCATIONS, POLLUTANTS, WEATHER_VARIABLES, TRAFFIC_DATA_FILE

# Data acquisition and processing imports
from data_fetcher import get_historical_data
from weather_fetcher import fetch_weather_data
from data_processor import process_air_quality_data
from traffic_data_loader import load_traffic_data, transform_traffic_data, filter_traffic_by_location, aggregate_hourly_traffic

# Analysis model imports
from models.traffic_aqi_analysis import traffic_aqi_correlation_analysis
from models.lagged_analysis import perform_lagged_analysis
from models.weather_correlation_and_mlr import correlation_analysis, multiple_linear_regression
from models.temporal_analysis import temporal_pattern_analysis
from models.spatial_analysis import spatial_comparison_analysis
from models.sarima_analysis import sarima_modeling
from models.pollutant_weather_analysis import analyze_individual_pollutants

# ----- Future model imports (currently commented) -----
# from models.random_forest import random_forest_analysis  # To be implemented
# from models.xgboost_model import xgboost_analysis  # To be implemented
# from models.prophet_forecast import prophet_forecast  # To be implemented
# from models.lstm_model import lstm_analysis  # To be implemented

def main():
    """
    Main function to execute the AirSense Copenhagen workflow.
    Follows a structured data science approach for comprehensive air quality analysis.
    """
    # Configure matplotlib to avoid warnings with many plots
    plt.rcParams['figure.max_open_warning'] = 50
    
    # Define location for analysis
    primary_location = "Torvegade"  # Selected urban canyon location
    
    print("\n" + "="*80)
    print("AIRSENSE COPENHAGEN: COMPREHENSIVE AIR QUALITY ANALYSIS")
    print("="*80)
    
    # =============================================================================
    # SECTION 1: DATA ACQUISITION & PREPARATION
    # =============================================================================
    print("\n--- DATA ACQUISITION & PREPARATION ---")
    
    # Fetch current air quality data for all locations
    location_dfs = get_historical_data(LOCATIONS, POLLUTANTS)
    
    # Process data and calculate AQI for all locations
    processed_dfs = process_air_quality_data(location_dfs)
    
    # Determine date range from primary location
    aqi_start_date = None
    aqi_end_date = None
    for name, df in processed_dfs.items():
        if name == primary_location and len(df) > 0:
            # Ensure AQI data is sorted chronologically
            df = df.sort_values(by='time', ascending=True)
            processed_dfs[name] = df  # Save the sorted dataframe
            
            aqi_start_date = df['time'].min().strftime('%Y-%m-%d')
            aqi_end_date = df['time'].max().strftime('%Y-%m-%d')
            print(f"AQI data spans: {aqi_start_date} to {aqi_end_date}")
    
    # Fetch weather data for primary location
    latitude, longitude = LOCATIONS[primary_location]
    df_weather = fetch_weather_data(latitude, longitude, 
                                   start_date=aqi_start_date, 
                                   end_date=aqi_end_date)
    
    # Check and handle data quality issues
    if df_weather is not None:
        # Sort chronologically
        df_weather = df_weather.sort_values(by='time', ascending=True)
        
        # Remove duplicate timestamps if any
        duplicate_count = df_weather.duplicated(subset=['time']).sum()
        if duplicate_count > 0:
            print(f"WARNING: Found {duplicate_count} duplicate timestamps in weather data")
            df_weather = df_weather.drop_duplicates(subset=['time'], keep='first')
        
        # Handle missing values with interpolation
        missing_vals = df_weather.isna().sum()
        if any(missing_vals > 0):
            print("\nMissing values in weather data:")
            print(missing_vals[missing_vals > 0])
            
            # Use time-based interpolation
            df_weather = df_weather.set_index('time')
            for col in df_weather.columns:
                df_weather[col] = df_weather[col].interpolate(method='time')
            df_weather = df_weather.reset_index()
            
            # Handle any remaining NaNs
            missing_after = df_weather.isna().sum()
            if any(missing_after > 0):
                print("Values still missing after interpolation (will use ffill/bfill):")
                print(missing_after[missing_after > 0])
                df_weather = df_weather.ffill().bfill()
    else:
        print(f"Failed to retrieve weather data for {primary_location}")
        return None
    
    # Data integration: Merge weather and AQI data
    merged_dfs = {}
    if primary_location in processed_dfs:
        # Get the AQI data
        aqi_df = processed_dfs[primary_location]
        
        # Remove duplicate timestamps if any
        duplicate_count = aqi_df.duplicated(subset=['time']).sum()
        if duplicate_count > 0:
            print(f"WARNING: Found {duplicate_count} duplicate timestamps in AQI data")
            aqi_df = aqi_df.drop_duplicates(subset=['time'], keep='first')
        
        # Merge datasets
        merged_df = pd.merge(
            aqi_df,
            df_weather,
            on='time',
            how='inner'
        )
        
        # Check merge results
        merge_count = len(merged_df)
        if merge_count > 0:
            # Sort the merged data
            merged_df = merged_df.sort_values(by='time', ascending=True)
            print(f"Successfully merged data with {merge_count} matching timestamps")
            
            # Check for NaN values in critical columns
            weather_cols = [col for col in WEATHER_VARIABLES if col in merged_df.columns]
            nan_check = merged_df[['AQI'] + weather_cols].isna().sum()
            if any(nan_check > 0):
                print("\nWarning: Merged data contains NaN values:")
                print(nan_check[nan_check > 0])
            
            merged_dfs[primary_location] = merged_df
        else:
            print("ERROR: Merge resulted in 0 matching rows - no overlapping timestamps!")
            print("Check that your AQI and weather data cover the same time period.")
            return None
    
    # =============================================================================
    # SECTION 2: EXPLORATORY DATA ANALYSIS
    # =============================================================================
    print("\n--- EXPLORATORY DATA ANALYSIS ---")
    
    # Basic descriptive statistics would be added here
    
    # Data profiling report placeholder
    print("\nGenerating data profile reports...")
    # TODO: Implement data profiling functionality
    # data_profile = generate_data_profile(merged_dfs[primary_location])
    
    # =============================================================================
    # SECTION 3: SINGLE-POLLUTANT ANALYSIS
    # =============================================================================
    print("\n--- SINGLE-POLLUTANT ANALYSIS ---")
    
    # Check if pollutant columns exist in the merged data
    if primary_location in merged_dfs and len(merged_dfs[primary_location]) >= 100:
        pollutant_list = POLLUTANTS.split(',') if isinstance(POLLUTANTS, str) else POLLUTANTS
        available_pollutants = [p for p in pollutant_list if p in merged_dfs[primary_location].columns]
        
        if available_pollutants:
            print("\nPerforming individual pollutant analysis...")
            pollutant_results = analyze_individual_pollutants(
                merged_dfs[primary_location],
                weather_variables=WEATHER_VARIABLES,
                output_dir="figures/pollutant_analysis"
            )
            
            # Find the best individual pollutant
            best_pollutant = None
            best_r2 = 0
            
            for pollutant, result in pollutant_results.items():
                if 'multiple_regression' in result and result['multiple_regression'].get('r2') is not None:
                    r2 = result['multiple_regression']['r2']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_pollutant = pollutant
            
            if best_pollutant and best_pollutant != 'AQI':
                print(f"\nBest individual pollutant: {best_pollutant} with R² = {best_r2:.4f}")
                print(f"Compared to AQI R² = {pollutant_results['AQI']['multiple_regression']['r2']:.4f}")
                
                if best_r2 > pollutant_results['AQI']['multiple_regression']['r2']:
                    print(f"Recommendation: Consider focusing on {best_pollutant} for improved modeling results")
                    improvement = (best_r2 - pollutant_results['AQI']['multiple_regression']['r2']) / pollutant_results['AQI']['multiple_regression']['r2'] * 100
                    print(f"Improvement potential: {improvement:.1f}% higher R²")
        else:
            print("No individual pollutant columns found in the dataset.")
    
    # PCA Analysis placeholder
    print("\nPrincipal Component Analysis of pollutants...")
    # TODO: Implement PCA analysis
    # pca_results = perform_pca_analysis(merged_dfs[primary_location], pollutant_list)
    
    # =============================================================================
    # SECTION 4: TEMPORAL ANALYSIS
    # =============================================================================
    print("\n--- TEMPORAL ANALYSIS ---")
    
    # Perform temporal pattern analysis
    print("\nPerforming temporal pattern analysis...")
    temporal_pattern_analysis(processed_dfs)
    
    # Perform lagged analysis for primary location
    if primary_location in merged_dfs and len(merged_dfs[primary_location]) >= 100:
        print("\nPerforming lagged analysis to examine temporal effects...")
        lagged_results = perform_lagged_analysis(
            merged_dfs[primary_location], 
            target_col='AQI', 
            feature_col='traffic_count',
            output_dir="figures/traffic_aqi_2014/lagged"
        )

        if lagged_results and 'model_results' in lagged_results:
            model, rmse, r2, feature_importance = lagged_results['model_results']
            print(f"\nLagged model performance metrics:")
            print(f"RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # Perform SARIMA modeling
    print("\nPerforming SARIMA modeling and forecasting...")
    sarima_modeling(processed_dfs)
    
    # Prophet Forecasting placeholder
    print("\nProphet forecasting model (placeholder)...")
    # TODO: Implement Prophet forecasting
    # prophet_results = prophet_forecast(merged_dfs[primary_location], target_col='AQI')
    
    # =============================================================================
    # SECTION 5: SPATIAL ANALYSIS
    # =============================================================================
    print("\n--- SPATIAL ANALYSIS ---")
    
    # Perform spatial comparison analysis
    print("\nPerforming spatial comparison analysis...")
    spatial_comparison_analysis(processed_dfs)
    
    # GIS Visualization placeholder
    print("\nGIS visualization (placeholder)...")
    # TODO: Implement GIS visualization
    # gis_vis = create_gis_visualization(processed_dfs, LOCATIONS)
    
    # =============================================================================
    # SECTION 6: ADVANCED MODELING
    # =============================================================================
    print("\n--- ADVANCED MODELING ---")
    
    # Weather and AQI correlation analysis and MLR modeling
    if primary_location in merged_dfs and len(merged_dfs[primary_location]) >= 100:
        # Basic correlation analysis
        print("\nPerforming weather-AQI correlation analysis...")
        corr, lag_corrs = correlation_analysis(merged_dfs[primary_location], 
                                             output_dir="figures/weather_aqi")
        
        if corr is not None:
            # Multiple linear regression (baseline model)
            print("\nBuilding multiple linear regression model...")
            model, rmse, r2, coef_df = multiple_linear_regression(merged_dfs[primary_location], 
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
    else:
        print(f"\nWARNING: Only {len(merged_dfs[primary_location]) if primary_location in merged_dfs else 0} data points available.")
        print("Insufficient data for reliable statistical analysis.")
    
    # Random Forest placeholder
    print("\nRandom Forest modeling (placeholder)...")
    # TODO: Implement Random Forest modeling
    # rf_results = random_forest_analysis(merged_dfs[primary_location], 
    #                                   target_col='AQI',
    #                                   output_dir="figures/advanced_models")
    
    # XGBoost placeholder
    print("\nXGBoost modeling (placeholder)...")
    # TODO: Implement XGBoost modeling
    # xgb_results = xgboost_analysis(merged_dfs[primary_location],
    #                               target_col='AQI',
    #                               output_dir="figures/advanced_models")
    
    # LSTM Neural Network placeholder
    print("\nLSTM Neural Network (placeholder)...")
    # TODO: Implement LSTM modeling
    # lstm_results = lstm_analysis(merged_dfs[primary_location],
    #                             target_col='AQI',
    #                             output_dir="figures/advanced_models")
    
    # =============================================================================
    # SECTION 7: BUSINESS INTELLIGENCE INSIGHTS
    # =============================================================================
    print("\n--- BUSINESS INTELLIGENCE INSIGHTS ---")
    
    # Key findings summary placeholder
    print("\nGenerating key findings summary...")
    # TODO: Implement key findings summary
    # key_findings = generate_key_findings(model_results, pollutant_results)
    
    # Decision support metrics placeholder
    print("\nComputing decision support metrics...")
    # TODO: Implement decision support metrics
    # decision_metrics = compute_decision_metrics(merged_dfs[primary_location])
    
    # Scenario analysis placeholder
    print("\nPerforming scenario analysis...")
    # TODO: Implement scenario analysis
    # scenarios = perform_scenario_analysis(model, merged_dfs[primary_location])
    
    # Interactive dashboard concept placeholder
    print("\nInteractive dashboard concept (placeholder)...")
    # TODO: Implement dashboard concept
    # dashboard_spec = generate_dashboard_specification(processed_dfs)
    
    # =============================================================================
    # SECTION 8: HISTORICAL VALIDATION
    # =============================================================================
    print("\n--- HISTORICAL VALIDATION ---")
    
    # Analyze historical data (2014)
    analyze_historical_2014_data()
    
    # Model consistency check placeholder
    print("\nPerforming model consistency check...")
    # TODO: Implement model consistency check
    # consistency_check = check_model_consistency(current_model, historical_model)
    
    # Change point analysis placeholder
    print("\nChange point analysis (placeholder)...")
    # TODO: Implement change point analysis
    # change_points = detect_change_points(merged_dfs[primary_location])
    
    print("\nAnalysis complete. Results saved to figures/ directory.")
    
    return merged_dfs

def analyze_historical_2014_data():
    """
    Analyze historical 2014 traffic data with matching air quality data.
    Used for validation and comparison with current data analysis.
    """
    print("\n" + "="*80)
    print("HISTORICAL ANALYSIS: 2014 TRAFFIC AND AIR QUALITY")
    print("="*80)
    
    # Define location to use
    location_name = "Torvegade"
    
    # Step 1: Load and process traffic data
    raw_traffic_df = load_traffic_data(TRAFFIC_DATA_FILE)
    processed_traffic_df = transform_traffic_data(raw_traffic_df)
    
    if processed_traffic_df is not None:
        # Step 2: Aggregate hourly traffic data
        aggregated_traffic = aggregate_hourly_traffic(processed_traffic_df)
        
        # Step 3: Filter for target location data
        location_traffic_df = filter_traffic_by_location(aggregated_traffic, location_name)
        
        if len(location_traffic_df) == 0:
            print(f"Error: No traffic data found for {location_name}")
            return
        
        print(f"\nUsing traffic data from {location_name}")
        location_coords = LOCATIONS[location_name]
        
        # Step 4: Create date range for air quality data (from 2014)
        min_date = location_traffic_df['datetime'].min().strftime('%Y-%m-%d')
        max_date = location_traffic_df['datetime'].max().strftime('%Y-%m-%d')
        
        # Step 5: Define the single location for air quality data
        historical_location = {
            location_name: location_coords
        }
        
        # Step 6: Fetch historical air quality data
        air_quality_dfs = get_historical_data(historical_location, POLLUTANTS, 
                                           start_date=min_date, end_date=max_date)
        
        # Step 7: Process air quality data to calculate AQI
        processed_aqi_dfs = process_air_quality_data(air_quality_dfs)
        
        # Get the AQI dataframe for our location
        if location_name in processed_aqi_dfs:
            aqi_df = processed_aqi_dfs[location_name]
            
            # Convert datetime to time if needed for merging
            if 'datetime' in location_traffic_df.columns and 'time' in aqi_df.columns:
                location_traffic_df = location_traffic_df.rename(columns={'datetime': 'time'})
            
            # Step 8: Merge traffic and AQI data
            merged_df = pd.merge(
                location_traffic_df,
                aqi_df,
                on='time',
                how='inner'
            )
            
            print(f"Successfully merged historical data: {len(merged_df)} records")
            
            if len(merged_df) > 0:
                # Perform traffic-AQI correlation analysis
                print("\nPerforming traffic-AQI correlation analysis...")
                correlation_results = traffic_aqi_correlation_analysis(merged_df, output_dir="figures/traffic_aqi_2014")
                
                if correlation_results and 'pearson' in correlation_results:
                    pearson_corr, pearson_p = correlation_results['pearson']
                    if not np.isnan(pearson_corr):
                        print(f"\nOverall Traffic-AQI Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
                
                # Analyze relationship between 2014 traffic and AQI with traffic variables only
                if len(merged_df) >= 100:
                    print("\nBuilding traffic-AQI regression model for 2014...")
                    
                    # Use only traffic-related variables
                    traffic_cols = ['traffic_count']
                    if 'entry_count' in merged_df.columns:
                        traffic_cols.append('entry_count')
                    
                    # Prepare model dataframe
                    model_df = merged_df[['AQI'] + traffic_cols].dropna()
                    
                    # Only proceed if we have valid data
                    if len(model_df) > 50:
                        model, rmse, r2, coef_df = multiple_linear_regression(model_df, 
                                                                          output_dir="figures/traffic_aqi_2014")
                        if model is not None:
                            print(f"\n2014 Traffic-AQI MLR Results:")
                            print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")
                            print("\nTraffic variable importance for 2014 data:")
                            print(coef_df)
            else:
                print("No matching records found between 2014 traffic and air quality data")
        else:
            print(f"No air quality data processed for {location_name}")
    
    print("\n2014 Historical analysis complete.")

if __name__ == "__main__":
    merged_dfs = main()
