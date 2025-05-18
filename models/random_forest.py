# models/random_forest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def random_forest_analysis(df, target_col='ozone', features=None, include_traffic=True, 
                           tune_hyperparams=True, output_dir="figures/random_forest"):
    """
    Perform Random Forest analysis to predict air pollutant concentrations.
    
    Parameters:
    df: DataFrame with weather, traffic, and pollutant data
    target_col: Target pollutant to predict (default: 'ozone')
    features: List of feature columns to use (if None, auto-selects weather and traffic)
    include_traffic: Whether to include traffic data as feature
    tune_hyperparams: Whether to perform hyperparameter tuning
    output_dir: Directory to save output figures
    
    Returns:
    Dictionary with model, metrics, and feature importance
    """
    print("\n" + "="*80)
    print(f"RANDOM FOREST MODELING: PREDICTING {target_col.upper()}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date for filenames
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Check if target column exists
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset")
        return None
    
    # Identify available weather features
    weather_features = [col for col in df.columns if col in [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'windspeed_10m', 'pressure_msl', 'winddirection_10m'
    ]]
    
    # Identify traffic features if requested
    traffic_features = []
    if include_traffic:
        traffic_features = [col for col in df.columns if col in ['traffic_count']]
        if not traffic_features:
            print("Warning: No traffic features found but include_traffic=True")
    
    # If features not specified, use weather and traffic features
    if features is None:
        features = weather_features + traffic_features
    
    print(f"\nTarget variable: {target_col}")
    print(f"Using {len(features)} features: {', '.join(features)}")
    
    # Check if we have sufficient features
    if len(features) < 2:
        print("Error: Not enough features for modeling")
        return None
    
    # Prepare data
    X = df[features].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    print("\nChecking for missing values...")
    missing_X = X.isna().sum().sum()
    missing_y = y.isna().sum()
    
    if missing_X > 0:
        print(f"Filling {missing_X} missing feature values...")
        # Random Forest can handle missing values, but we'll fill them for consistency
        X = X.fillna(X.mean())
    
    if missing_y > 0:
        print(f"Removing {missing_y} rows with missing target values...")
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
    
    print(f"\nFinal dataset size: {len(X)} rows")
    if len(X) < 100:
        print("Warning: Small dataset size may affect model performance")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Train Random Forest model
    if tune_hyperparams:
        print("\nPerforming hyperparameter tuning with GridSearchCV...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        model = RandomForestRegressor(random_state=42)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5,
            scoring='neg_mean_squared_error',
            verbose=0, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        model = grid_search.best_estimator_
    else:
        print("\nTraining Random Forest model with default parameters...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_ev = explained_variance_score(y_test, y_pred_test)
    
    print("\nModel Performance:")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Testing MAE: {test_mae:.2f}")
    print(f"Testing Explained Variance: {test_ev:.4f}")
    
    # Check for overfitting
    if train_r2 - test_r2 > 0.2:
        print("Warning: Potential overfitting detected")
    
    # Calculate feature importance
    importance_dict = {}
    for i, feature in enumerate(features):
        importance_dict[feature] = model.feature_importances_[i]
    
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    for index, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Random Forest Feature Importance for {target_col} Prediction')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_{target_col}_feature_importance.png", dpi=300)
    plt.close()
    
    # Create scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'Random Forest Predictions vs Actual Values (R²={test_r2:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_{target_col}_predictions.png", dpi=300)
    plt.close()
    
    # Plot feature effect for the top 3 features
    top_features = importance_df['Feature'].head(3).tolist()
    
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        
        # Create partial dependence plot (simplified)
        # Sort the feature values and predictions to see the trend
        feature_values = X_test[feature].values
        predictions = y_pred_test
        
        # Sort by feature value
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_preds = predictions[sorted_indices]
        
        # Plot individual points
        plt.scatter(sorted_values, sorted_preds, alpha=0.3, label='Predictions')
        
        # Plot a smoothed trend line
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(sorted_preds, sigma=10)
        plt.plot(sorted_values, smoothed, 'r-', linewidth=2, label='Trend')
        
        plt.xlabel(feature)
        plt.ylabel(f'Predicted {target_col}')
        plt.title(f'Effect of {feature} on {target_col} Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{date_str}_{target_col}_{feature}_effect.png", dpi=300)
        plt.close()
    
    # Save results
    results = {
        'model': model,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_ev': test_ev
        },
        'feature_importance': importance_df,
        'features': features,
        'target': target_col
    }
    
    return results

def compare_pollutant_rf_models(merged_df, pollutant_list, include_traffic=True, historical=False, output_dir="figures/random_forest_comparison"):
    """
    Compare Random Forest models for each pollutant and AQI.
    
    Parameters:
    merged_df: DataFrame containing pollutants, weather, and possibly traffic data
    pollutant_list: List of pollutants to analyze
    include_traffic: Whether to include traffic data as features
    historical: Whether this is historical (2014) data
    output_dir: Directory to save output files
    
    Returns:
    DataFrame with model performance metrics for each pollutant
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure pollutant_list is a list (not a comma-separated string)
    if isinstance(pollutant_list, str):
        pollutant_list = pollutant_list.split(',')
    
    # Add AQI to the list
    pollutant_list = pollutant_list + ['AQI']
    
    # Get weather columns
    weather_cols = [col for col in merged_df.columns if col in [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'windspeed_10m', 'pressure_msl', 'winddirection_10m'
    ]]
    
    # Get traffic columns if requested
    traffic_cols = []
    if include_traffic and 'traffic_count' in merged_df.columns:
        traffic_cols = ['traffic_count']
    
    # Combined features
    all_features = weather_cols + traffic_cols
    
    # Check if we have sufficient features
    if len(all_features) < 2:
        print("Error: Not enough features for modeling")
        return None
    
    # Prepare results table
    results = []
    
    for pollutant in pollutant_list:
        if pollutant not in merged_df.columns:
            print(f"Skipping {pollutant} - not found in dataset")
            continue
        
        print(f"\n{'-' * 20}")
        print(f"Processing Random Forest model for {pollutant}...")
        
        # Keep original data structure
        output_subdir = f"{output_dir}/{'historical' if historical else 'current'}/{pollutant}"
        
        try:
            # Run Random Forest for this pollutant
            model_results = random_forest_analysis(
                merged_df,
                target_col=pollutant,
                features=all_features,
                include_traffic=include_traffic,
                tune_hyperparams=True,
                output_dir=output_subdir
            )
            
            if model_results is not None:
                # Extract performance metrics
                test_r2 = model_results['metrics']['test_r2']
                test_rmse = model_results['metrics']['test_rmse']
                
                # Get top features (top 3)
                top_features = []
                for i, (_, row) in enumerate(model_results['feature_importance'].iterrows()):
                    if i < 3:
                        top_features.append(f"{row['Feature']} ({row['Importance']:.3f})")
                
                # Add to results table
                results.append({
                    'Pollutant': pollutant,
                    'Historical': historical,
                    'Model': 'Random Forest',
                    'R²': test_r2,
                    'RMSE': test_rmse,
                    'Top Features': ', '.join(top_features)
                })
        except Exception as e:
            print(f"Error processing {pollutant}: {e}")
    
    if not results:
        print("No valid models were created")
        return None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    period = 'historical' if historical else 'current'
    results_df.to_csv(f"{output_dir}/{period}_rf_model_comparison.csv", index=False)
    
    # Print summary table
    print(f"\n{'-' * 60}")
    print(f"{'Historical' if historical else 'Current'} Period - Random Forest Model Comparison")
    print(f"{'-' * 60}")
    print(results_df[['Pollutant', 'R²', 'RMSE']].sort_values('R²', ascending=False))
    
    # Create visualization of R² values
    plt.figure(figsize=(10, 6))
    chart_df = results_df.sort_values('R²', ascending=False)
    bars = plt.bar(chart_df['Pollutant'], chart_df['R²'])
    
    # Color the bars
    for i, bar in enumerate(bars):
        if chart_df['R²'].iloc[i] > 0.7:
            bar.set_color('green')
        elif chart_df['R²'].iloc[i] > 0.4:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)  # Reference line at R²=0.5
    plt.title(f"Random Forest Performance (R²) by Pollutant - {'Historical' if historical else 'Current'} Period")
    plt.ylabel('R² (Test Data)')
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{period}_r2_comparison.png", dpi=300)
    plt.close()
    
    return results_df
