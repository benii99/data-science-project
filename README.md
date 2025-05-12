# AirSense Copenhagen

## Project Overview

AirSense Copenhagen is a data science project that leverages real-time air pollution data to identify pollution hotspots in Copenhagen's urban areas. The system integrates air quality measurements across different locations to create insights that can inform targeted traffic management solutions.

This project supports Copenhagen's carbon neutrality goals by providing data-driven tools for analyzing the relationship between traffic patterns and air pollution levels. The implementation calculates a scientifically valid Air Quality Index (AQI) according to EPA standards, allowing for consistent assessment of air quality across different urban environments.

## Features

- Retrieves air quality data from Open-Meteo's Air Quality API
- Collects data for multiple key locations in Copenhagen
- Implements the EPA's Air Quality Index calculation methodology
- Uses proper averaging periods for different pollutants:
  - 24-hour averages for PM2.5 and PM10
  - 8-hour averages for O3 and CO
  - 1-hour values for NO2 and SO2
- Identifies dominant pollutants at each location and timestamp
- Categorizes air quality according to standard EPA categories

## Project Structure

```
airsense_copenhagen/
├── config.py             # Configuration settings and constants
├── aqi_breakpoints.py    # EPA AQI breakpoints for different pollutants
├── aqi_calculator.py     # Functions for calculating AQI values
├── data_fetcher.py       # API interaction and data retrieval
├── data_processor.py     # Data processing and AQI calculation pipeline
├── main.py               # Main script to execute the workflow
└── README.md             # Project documentation
```

### File Descriptions

- **config.py**: Contains location coordinates, API settings, and conversion factors
- **aqi_breakpoints.py**: Defines EPA breakpoints for different pollutants
- **aqi_calculator.py**: Functions to calculate AQI values from pollutant concentrations
- **data_fetcher.py**: Handles API requests to retrieve air quality data
- **data_processor.py**: Processes raw data to calculate rolling averages and AQI
- **main.py**: Orchestrates the entire workflow

## Installation

1. Clone the repository:
   ```
   git clone [GITHUB URL]
   cd [PROJECT NAME]
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows, use: env\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install requests pandas numpy
   ```

## Usage

Run the main script to collect and process air quality data:

```
python main.py
```

This will:
1. Download air quality data for the past year from three distinct locations in Copenhagen
2. Calculate rolling averages based on EPA standards
3. Compute AQI values for each timestamp
4. Display the most recent results for each location

### Working with the Results

The processed data is stored in DataFrames with the following key columns:

- `time`: Timestamp for the measurements
- `AQI`: Calculated Air Quality Index value
- `AQI_Category`: Text category (Good, Moderate, etc.)
- `Dominant_Pollutant`: The pollutant responsible for the AQI value

To work with specific locations or time periods:

```python
import pandas as pd
from main import main

# Get processed data
processed_dfs = main()

# Access a specific location
hcab_data = processed_dfs["H.C. Andersens Boulevard"]

# Filter for a specific time period
last_month = hcab_data[hcab_data['time'] > pd.Timestamp.now() - pd.Timedelta(days=30)]

# Calculate average AQI for this period
avg_aqi = last_month['AQI'].mean()
print(f"Average AQI for last month: {avg_aqi:.1f}")
```

## Technical Approach

### Data Sources

The project uses the Open-Meteo Air Quality API, which provides historical and current air quality data with 1-2km spatial resolution. The API offers hourly measurements for multiple pollutants:

- Particulate Matter (PM2.5 and PM10)
- Nitrogen Dioxide (NO2)
- Carbon Monoxide (CO)
- Sulfur Dioxide (SO2)
- Ozone (O3)

### Locations

We analyze three strategic locations that represent different urban environments:

1. **H.C. Andersens Boulevard**: High-traffic urban center with commercial activity
2. **Nørrebro Residential Area**: Medium-density mixed-use neighborhood
3. **Amager Strandpark**: Coastal recreational area with minimal traffic

### AQI Calculation Methodology

The Air Quality Index calculation follows the EPA standard methodology:

1. Calculate appropriate rolling averages for each pollutant
2. Convert concentration values to a standardized index scale (0-500)
3. Determine overall AQI as the maximum of individual pollutant indices
4. Identify the dominant pollutant responsible for the AQI value
5. Categorize AQI values according to health impact levels

## Future Development

Potential enhancements for this project include:

- Integration with traffic density data
- Predictive modeling of air quality based on traffic patterns
- Visualization dashboard for real-time monitoring
- Expansion to additional locations in Copenhagen
- Temporal analysis to identify daily/weekly/seasonal patterns

## Acknowledgments

- Open-Meteo for providing the Air Quality API
- EPA for the AQI calculation methodology and breakpoints
