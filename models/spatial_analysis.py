import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os
from config import LOCATIONS
import geopandas as gpd
from shapely.geometry import Point

def spatial_comparison_analysis(processed_dfs, output_dir="figures/"):
    """
    Perform spatial comparison analysis between different locations.
    
    Parameters:
    processed_dfs: Dictionary of location names and processed DataFrames
    output_dir: Directory to save output figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prepare data for comparison
    location_series = {}
    for location, df in processed_dfs.items():
        # Ensure data is sorted by time and set index
        df = df.sort_values('time')
        location_series[location] = df.set_index('time')['AQI']
    
    # Create a combined DataFrame
    combined_df = pd.DataFrame(location_series)
    combined_df = combined_df.dropna()
    
    # 2. Correlation analysis
    correlation = combined_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Between Locations')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/location_correlation.png", dpi=300)
    plt.close()
    
    # 3. Compare daily patterns between locations
    daily_patterns = {}
    for location, df in processed_dfs.items():
        hourly_avg = df.groupby(df['time'].dt.hour)['AQI'].mean()
        daily_patterns[location] = hourly_avg
    
    daily_patterns_df = pd.DataFrame(daily_patterns)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_patterns_df)
    plt.title('Average Daily AQI Pattern by Location')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average AQI')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/location_daily_patterns.png", dpi=300)
    plt.close()
    
    # 4. Create comparison table of key metrics across locations
    metrics = []
    for location, df in processed_dfs.items():
        metrics.append({
            'Location': location,
            'Mean AQI': df['AQI'].mean(),
            'Median AQI': df['AQI'].median(),
            'Max AQI': df['AQI'].max(),
            'Min AQI': df['AQI'].min(),
            'Std Dev': df['AQI'].std(),
            'Dominant Pollutant': df['Dominant_Pollutant'].mode()[0] if 'Dominant_Pollutant' in df.columns else 'N/A'
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.set_index('Location')
    metrics_df.to_csv(f"{output_dir}/location_metrics_comparison.csv")
    print("Saved location metrics comparison table.")
    
    # 5. Map visualization of locations with color-coded metrics
    try:
        # Create points from coordinates
        geometry = [Point(lon, lat) for location, (lat, lon) in LOCATIONS.items()]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                'Location': list(LOCATIONS.keys()),
                'Mean AQI': [metrics_df.loc[loc, 'Mean AQI'] if loc in metrics_df.index else np.nan 
                            for loc in LOCATIONS.keys()]
            },
            geometry=geometry,
            crs="EPSG:4326"
        )
        
        # Try to get a Denmark/Copenhagen base map
        try:
            # Get the Denmark outline
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            denmark = world[world.name == 'Denmark']
            
            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot Denmark in light gray
            denmark.plot(ax=ax, color='lightgray')
            
            # Define Copenhagen bounding box (approximate)
            lon_min, lon_max = 12.45, 12.70
            lat_min, lat_max = 55.60, 55.75
            
            # Set plot limits to zoom on Copenhagen
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            
            # Plot locations with color based on Mean AQI
            gdf.plot(ax=ax, column='Mean AQI', cmap='viridis', legend=True, 
                    markersize=200, alpha=0.7)
            
            # Add location labels
            for idx, row in gdf.iterrows():
                ax.annotate(row['Location'], xy=(row.geometry.x, row.geometry.y),
                           xytext=(3, 3), textcoords="offset points",
                           fontsize=12, fontweight='bold')
            
            plt.title('Air Quality Monitoring Locations in Copenhagen\n(Color indicates Mean AQI)')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/location_map.png", dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Could not create detailed map with base layer: {e}")
            
            # Fallback to simple scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            gdf.plot(ax=ax, column='Mean AQI', cmap='viridis', legend=True,
                   markersize=200)
            
            # Add location labels
            for idx, row in gdf.iterrows():
                ax.annotate(row['Location'], xy=(row.geometry.x, row.geometry.y),
                           xytext=(3, 3), textcoords="offset points")
            
            plt.title('Air Quality Monitoring Locations\n(Color indicates Mean AQI)')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/location_map.png", dpi=300)
            plt.close()
        
        print("Saved location map visualization.")
            
    except Exception as e:
        print(f"Error creating location map: {e}")
        print("You may need to install geopandas: pip install geopandas")
    
    print("Completed spatial comparison analysis")
