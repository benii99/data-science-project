import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import os
from datetime import datetime, timedelta

def create_lagged_features(df, target_col, feature_cols, lag_hours=[1, 2, 3, 6, 12, 24]):
    """
    Create lagged features for time series analysis.
    
    Parameters:
    df: DataFrame with time series data
    target_col: Name of the target column (e.g., 'AQI')
    feature_cols: List of feature column names to create lags for (e.g., ['traffic_count'])
    lag_hours: List of lag periods in hours
    
    Returns:
    DataFrame with additional lagged features
    """
    # Create a copy to avoid modifying the original
    df_lagged = df.copy()
    
    # Ensure df is sorted by time
    if 'time' in df_lagged.columns:
        time_col = 'time'
    elif 'datetime' in df_lagged.columns:
        time_col = 'datetime'
    else:
        print("Error: No time column found")
        return df_lagged
    
    df_lagged = df_lagged.sort_values(time_col)
    
    # Create lagged features
    for col in feature_cols:
        for lag in lag_hours:
            lag_name = f"{col}_lag_{lag}h"
            df_lagged[lag_name] = df_lagged[col].shift(lag)
    
    # Instead of dropping all NaN values, only drop rows with NaNs in target or lagged features
    cols_to_check = [target_col] + [f"{col}_lag_{lag}h" for col in feature_cols for lag in lag_hours]
    rows_before = len(df_lagged)
    df_lagged = df_lagged.dropna(subset=cols_to_check)
    rows_after = len(df_lagged)
    
    print(f"Created lagged features: Rows reduced from {rows_before} to {rows_after} due to lag creation")
    
    # Handle the case where all rows were dropped
    if rows_after == 0:
        print("WARNING: All rows were dropped when creating lagged features.")
        print("This can happen if there are too many NaN values or if the lag periods are too large.")
        
        # Return a subset with just a few lags to avoid complete data loss
        df_lagged = df.copy()
        df_lagged = df_lagged.sort_values(time_col)
        
        # Try with smaller lags
        small_lags = [1, 2, 3]
        print(f"Attempting with smaller lags: {small_lags}")
        for col in feature_cols:
            for lag in small_lags:
                lag_name = f"{col}_lag_{lag}h"
                df_lagged[lag_name] = df_lagged[col].shift(lag)
        
        # Only check essential columns
        essential_cols = [target_col] + [f"{col}_lag_{lag}h" for col in feature_cols for lag in small_lags]
        df_lagged = df_lagged.dropna(subset=essential_cols)
        print(f"After using smaller lags: {len(df_lagged)} rows remaining")
        
        if len(df_lagged) == 0:
            print("ERROR: Still no valid data after trying smaller lags.")
            # Return a special marker to indicate failure
            return None
    
    return df_lagged

def analyze_lag_correlations(df, target_col='AQI', feature_col='traffic_count', max_lag=24, output_dir="figures/lagged_analysis"):
    """
    Analyze correlations between a target variable and lagged versions of a feature.
    
    Parameters:
    df: DataFrame with time series data
    target_col: Name of target column (e.g., 'AQI')
    feature_col: Name of feature column to create lags for (e.g., 'traffic_count')
    max_lag: Maximum number of hours to lag (reduced from 48 to 24 for practicality)
    output_dir: Directory to save output figures
    
    Returns:
    DataFrame with correlation results for each lag
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date for filename
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Check for required columns
    if target_col not in df.columns or feature_col not in df.columns:
        print(f"Error: Required columns not found. Need {target_col} and {feature_col}")
        return None
    
    # Ensure data types are numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df[feature_col] = pd.to_numeric(df[feature_col], errors='coerce')
    
    # Ensure df is sorted by time
    if 'time' in df.columns:
        time_col = 'time'
    elif 'datetime' in df.columns:
        time_col = 'datetime'
    else:
        print("Error: No time column found")
        return None
    
    df = df.sort_values(time_col)
    
    # Calculate correlations for each lag
    correlations = []
    p_values = []
    lags = list(range(0, max_lag + 1))
    
    print(f"Calculating correlations for {len(lags)} lag periods...")
    
    for lag in lags:
        if lag == 0:
            # Current values (no lag)
            corr, p = pearsonr(df[feature_col], df[target_col])
        else:
            # Lagged values
            lagged_feature = df[feature_col].shift(lag)
            # Remove NaN rows
            valid_mask = ~lagged_feature.isna()
            if valid_mask.sum() > 10:  # Only calculate if we have at least 10 valid points
                corr, p = pearsonr(lagged_feature[valid_mask], df[target_col][valid_mask])
            else:
                corr, p = np.nan, np.nan
        
        correlations.append(corr)
        p_values.append(p)
    
    # Create a DataFrame with the results
    results = pd.DataFrame({
        'lag_hours': lags,
        'correlation': correlations,
        'p_value': p_values,
        'significant': [p <= 0.05 for p in p_values]
    })
    
    # Find the lag with the strongest correlation (ignoring NaNs)
    valid_results = results.dropna()
    if len(valid_results) > 0:
        best_lag = valid_results.loc[valid_results['correlation'].abs().idxmax()]
        print(f"Strongest correlation at lag {best_lag['lag_hours']} hours: {best_lag['correlation']:.4f} (p-value: {best_lag['p_value']:.4f})")
    else:
        print("No valid correlations found.")
        return results
    
    # Plot lag correlations
    plt.figure(figsize=(12, 6))
    plt.bar(results['lag_hours'], results['correlation'], color=[
        'red' if sig and corr < 0 else 'green' if sig and corr > 0 else 'gray' 
        for sig, corr in zip(results['significant'], results['correlation'])
    ])
    plt.xlabel('Lag (hours)')
    plt.ylabel(f'Correlation with {target_col}')
    plt.title(f'Correlation between {target_col} and Lagged {feature_col}')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add a rectangle highlighting the optimal lag
    plt.gca().add_patch(plt.Rectangle(
        (best_lag['lag_hours'] - 0.4, min(0, best_lag['correlation'] - 0.05)), 
        0.8, 
        abs(best_lag['correlation']) + 0.1, 
        fill=False, 
        edgecolor='blue',
        linewidth=2
    ))
    
    plt.xticks(lags[::4])  # Show every 4th lag for readability
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_lag_correlations.png", dpi=300)
    plt.close()
    
    return results

def build_lagged_regression_model(df, target_col='AQI', feature_col='traffic_count', 
                                  selected_lags=[1, 3, 6, 12, 24], 
                                  output_dir="figures/lagged_analysis"):
    """
    Build a regression model using lagged features.
    
    Parameters:
    df: DataFrame with time series data
    target_col: Name of target column (e.g., 'AQI')
    feature_col: Name of feature column to create lags for (e.g., 'traffic_count')
    selected_lags: List of lag hours to use in the model
    output_dir: Directory to save output figures
    
    Returns:
    Tuple of (model, rmse, r2, feature_importance_df)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date for filename
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Create lagged features
    df_lagged = create_lagged_features(
        df, 
        target_col=target_col, 
        feature_cols=[feature_col], 
        lag_hours=selected_lags
    )
    
    # Check if we have valid data
    if df_lagged is None or len(df_lagged) == 0:
        print("Error: No data available after creating lagged features")
        return None, None, None, None
    
    # Prepare feature matrix
    feature_names = [f"{feature_col}_lag_{lag}h" for lag in selected_lags]
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_names if col not in df_lagged.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Only use columns that exist
        feature_names = [col for col in feature_names if col in df_lagged.columns]
        
    if not feature_names:
        print("Error: No valid feature columns available")
        return None, None, None, None
    
    X = df_lagged[feature_names]
    y = df_lagged[target_col]
    
    print(f"Final dataset shape for regression: {X.shape}")
    
    # Split data, respecting time series nature
    if len(X) > 50:  # Make sure we have enough data to split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    else:
        # Use all data for both training and testing if we have too few samples
        print("Warning: Limited data available, using same data for training and testing")
        X_train, X_test = X, X
        y_train, y_test = y, y
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    feature_importance = feature_importance.sort_values(by='Coefficient', key=abs, ascending=False)
    
    # Print results
    print(f"\nLagged Regression Model Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'Lagged Model: Predicted vs Actual {target_col} (RMSE={rmse:.2f}, R²={r2:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_lagged_predicted_vs_actual.png", dpi=300)
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
    plt.title('Lagged Feature Importance')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_lagged_feature_importance.png", dpi=300)
    plt.close()
    
    return model, rmse, r2, feature_importance

def perform_lagged_analysis(df, target_col='AQI', feature_col='traffic_count', output_dir="figures/lagged_analysis"):
    """
    Perform comprehensive lagged analysis of the relationship between traffic and AQI.
    
    Parameters:
    df: DataFrame with time series data
    target_col: Name of target column (e.g., 'AQI')
    feature_col: Name of feature column to analyze (e.g., 'traffic_count')
    output_dir: Directory to save output figures
    
    Returns:
    Dictionary of analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("LAGGED ANALYSIS: TRAFFIC AND AIR QUALITY")
    print("="*50)
    
    # Verify input data
    if target_col not in df.columns or feature_col not in df.columns:
        print(f"Error: Required columns not found in dataset. Need {target_col} and {feature_col}")
        return None
    
    # Check for time column
    if 'time' not in df.columns and 'datetime' not in df.columns:
        print("Error: No time column found in dataset")
        return None
        
    # Make a clean copy and convert to numeric
    df_clean = df.copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean[feature_col] = pd.to_numeric(df_clean[feature_col], errors='coerce')
    
    # Drop rows with NaN in key columns
    original_len = len(df_clean)
    df_clean = df_clean.dropna(subset=[target_col, feature_col])
    dropped = original_len - len(df_clean)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaN values in {target_col} or {feature_col}")
    
    # Step 1: Analyze correlations at different lags - use shorter max_lag
    print("\nAnalyzing correlations at different lag times...")
    lag_correlations = analyze_lag_correlations(
        df_clean, 
        target_col=target_col, 
        feature_col=feature_col, 
        max_lag=12,  # Reduced from 24 to avoid data loss
        output_dir=output_dir
    )
    
    if lag_correlations is None or lag_correlations.empty:
        print("Error: Could not calculate lag correlations")
        return None
    
    # Identify significant lags
    significant_lags = lag_correlations[lag_correlations['significant']]['lag_hours'].tolist()
    
    # If we found significant lags, use them; otherwise use defaults
    selected_lags = None
    if significant_lags:
        # Select a subset of significant lags (max 5)
        if len(significant_lags) > 5:
            selected_lags = sorted(significant_lags[:5])
        else:
            selected_lags = sorted(significant_lags)
            
        print(f"\nUsing {len(selected_lags)} significant lags: {selected_lags}")
    else:
        # Use small lags for safety
        selected_lags = [1, 2, 3]
        print(f"\nNo significant lags found. Using small default lags: {selected_lags}")
    
    # Step 2: Build lagged regression model
    print("\nBuilding regression model with lagged features...")
    model_results = build_lagged_regression_model(
        df_clean, 
        target_col=target_col, 
        feature_col=feature_col,
        selected_lags=selected_lags, 
        output_dir=output_dir
    )
    
    # Return results
    return {
        'lag_correlations': lag_correlations,
        'selected_lags': selected_lags,
        'model_results': model_results
    }
