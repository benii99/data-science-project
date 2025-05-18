import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import shap

def xgboost_analysis(df, target_col='ozone', features=None, include_traffic=True, 
                    tune_hyperparams=True, output_dir="figures/xgboost"):
    """
    Perform XGBoost analysis to predict air pollutant concentrations.
    
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
    print(f"XGBOOST MODELING: PREDICTING {target_col.upper()}")
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
        traffic_features = [col for col in df.columns if col in ['traffic_count', 'entry_count']]
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
        # Simple imputation for demonstration - in practice might use more sophisticated methods
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
    
    # Train XGBoost model
    if tune_hyperparams:
        print("\nPerforming hyperparameter tuning with GridSearchCV...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
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
        print("\nTraining XGBoost model with default parameters...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
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
    plt.title(f'XGBoost Feature Importance for {target_col} Prediction')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_{target_col}_feature_importance.png", dpi=300)
    
    # Create scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'XGBoost Predictions vs Actual Values (R²={test_r2:.4f})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{date_str}_{target_col}_predictions.png", dpi=300)
    
    # SHAP analysis for interpretability
    try:
        print("\nPerforming SHAP analysis for model interpretability...")
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance for {target_col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{date_str}_{target_col}_shap_importance.png", dpi=300)
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP Summary Plot for {target_col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{date_str}_{target_col}_shap_summary.png", dpi=300)
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        print("SHAP visualizations skipped. Install shap package if needed.")
    
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
