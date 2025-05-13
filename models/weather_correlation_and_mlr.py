import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from config import WEATHER_VARIABLES

def correlation_analysis(df, output_dir="figures/"):
    """
    Perform correlation analysis between weather variables and AQI.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Get current date for filename
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Use weather variables from config
    weather_vars = [var for var in WEATHER_VARIABLES if var in df.columns]
    
    if not weather_vars:
        print("No weather variables found in the dataset")
        return None, None
        
    # Select only AQI and weather variables
    analysis_df = df[["AQI"] + weather_vars].copy()
    
    # Check for and report missing values
    missing = analysis_df.isna().sum()
    if missing.sum() > 0:
        print("Missing values in analysis data:")
        print(missing[missing > 0])
        # Fill missing values with mean for numeric analysis
        analysis_df = analysis_df.fillna(analysis_df.mean())
    
    # Calculate correlation
    corr = analysis_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Matrix - Weather Variables vs AQI')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_weather_aqi_correlation_matrix.png", dpi=300)
    plt.close()  # Close figure to prevent warning

    # Lagged correlation for weather variables vs AQI
    lags = range(0, 25)  # 0 to 24 hours lag
    lag_corrs = {}
    
    for var in weather_vars:
        lag_corrs[var] = []
        for lag in lags:
            shifted = analysis_df[var].shift(lag)
            valid = analysis_df['AQI'].notna() & shifted.notna()
            if valid.sum() > 0:
                corr_val = analysis_df.loc[valid, 'AQI'].corr(shifted[valid])
            else:
                corr_val = np.nan
            lag_corrs[var].append(corr_val)

    plt.figure(figsize=(12, 6))
    for var in weather_vars:
        plt.plot(lags, lag_corrs[var], label=var)
    plt.xlabel('Lag (hours)')
    plt.ylabel('Correlation with AQI')
    plt.title('Time-Lagged Correlation of Weather Variables with AQI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_weather_aqi_lagged_correlation.png", dpi=300)
    plt.close()  # Close figure to prevent warning

    return corr, lag_corrs

def multiple_linear_regression(df, output_dir="figures/"):
    """
    Perform multiple linear regression (MLR) to predict AQI from weather variables.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date for filename
    date_str = datetime.now().strftime("%Y%m%d")

    # Use weather variables from config
    weather_vars = [var for var in WEATHER_VARIABLES if var in df.columns]
    
    if not weather_vars:
        print("No weather variables found in the dataset for MLR")
        return None, None, None, None
    
    # Prepare data
    df_clean = df.dropna(subset=['AQI'])
    
    # Handle missing values - drop rows with any NaN in weather variables
    df_clean = df_clean.dropna(subset=weather_vars)
    print(f"Data shape after selecting weather variables and removing NaNs: {df_clean.shape}")
    
    if len(df_clean) < 100:  # Arbitrary threshold for minimum data points
        print("WARNING: Not enough data points for reliable regression after removing NaNs")
    
    X = df_clean[weather_vars]
    y = df_clean['AQI']

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Plot predicted vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title(f'Weather-based AQI Prediction (RMSE={rmse:.2f}, R²={r2:.2f})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_weather_aqi_prediction_results.png", dpi=300)
    plt.close()  # Close figure to prevent warning

    # Plot coefficients
    coef_df = pd.DataFrame({'Variable': weather_vars, 'Coefficient': model.coef_})
    coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Variable', data=coef_df)
    plt.title('Weather Variable Importance for AQI Prediction')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_weather_variable_importance.png", dpi=300)
    plt.close()  # Close figure to prevent warning

    coef_df.to_csv(f"{output_dir}/{date_str}_mlr_coefficients.csv", index=False)

    return model, rmse, r2, coef_df
