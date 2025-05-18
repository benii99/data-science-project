# models/pollutant_weather_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from config import POLLUTANTS, WEATHER_VARIABLES

def analyze_individual_pollutants(merged_df, weather_variables=None, output_dir="figures/pollutant_analysis"):
    """
    Analyze the relationship between weather variables and each individual pollutant, plus AQI.
    
    Parameters:
    merged_df: DataFrame containing both weather data and pollutant data
    weather_variables: List of weather variables to analyze (defaults to WEATHER_VARIABLES from config)
    output_dir: Directory to save output figures
    
    Returns:
    Dictionary with results for each pollutant and AQI
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # If weather_variables not provided, use from config
    if weather_variables is None:
        weather_variables = WEATHER_VARIABLES
        
    # Extract pollutant list - convert string to list if needed
    pollutant_list = POLLUTANTS.split(',') if isinstance(POLLUTANTS, str) else POLLUTANTS
    
    # Add AQI to the analysis targets
    analysis_targets = pollutant_list + ['AQI']
    
    # Dictionary to store results
    results = {}
    
    print("\n" + "="*80)
    print("INDIVIDUAL POLLUTANT WEATHER CORRELATION ANALYSIS")
    print("="*80)
    
    # Check which targets are available in the data
    available_targets = [target for target in analysis_targets if target in merged_df.columns]
    missing_targets = [target for target in analysis_targets if target not in merged_df.columns]
    
    if missing_targets:
        print(f"Warning: The following targets are not available in the data: {', '.join(missing_targets)}")
    
    print(f"Analyzing correlations for: {', '.join(available_targets)}")
    
    # Create a comprehensive correlation table for all pollutants x weather variables
    print("\nGenerating comprehensive correlation table...")
    # Initialize with explicit float values
    correlation_matrix = pd.DataFrame(0.0, index=weather_variables, columns=available_targets)
    
    for target in available_targets:
        for var in weather_variables:
            if pd.notna(merged_df[target]).any() and pd.notna(merged_df[var]).any():
                try:
                    correlation = merged_df[[target, var]].dropna().corr().iloc[0, 1]
                    correlation_matrix.loc[var, target] = correlation
                except Exception as e:
                    print(f"Error calculating correlation between {var} and {target}: {e}")
                    correlation_matrix.loc[var, target] = np.nan
    
    # Ensure all values are float
    correlation_matrix = correlation_matrix.astype(float)
    
    # Replace any remaining NaN values with 0
    correlation_matrix = correlation_matrix.fillna(0)
    
    # Display and save the comprehensive correlation table
    print("\nComprehensive correlation table:")
    print(correlation_matrix.round(3))
    
    # Save correlation table to CSV
    correlation_matrix.to_csv(os.path.join(output_dir, "comprehensive_correlation_table.csv"))
    
    # Create a heatmap of the comprehensive correlation table
    try:
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Comprehensive Correlation: Weather Variables × Pollutants')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comprehensive_correlation_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
    
    # For each available target (pollutant or AQI)
    for target in available_targets:
        print(f"\nAnalyzing {target}...")
        target_results = {}
        
        # Only create detailed visualizations for ozone, nitrogen_dioxide, and AQI
        detailed_analysis = target in ['ozone', 'nitrogen_dioxide', 'AQI']
        
        # Calculate correlations between target and weather variables
        try:
            corr_data = merged_df[[target] + weather_variables].dropna().corr()
            target_correlations = corr_data.loc[target, weather_variables]
            
            # Store correlation values
            target_results['correlations'] = target_correlations.to_dict()
            
            # Print correlations
            print(f"Correlations with {target}:")
            for var, corr in target_correlations.items():
                print(f"  {var}: {corr:.4f}")
            
            # Create correlation heatmap only for selected pollutants
            if detailed_analysis:
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
                plt.title(f'Correlation between {target} and Weather Variables')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{target}_weather_correlation.png"))
                plt.close()
        except Exception as e:
            print(f"Error in correlation analysis for {target}: {e}")
            target_results['correlations'] = {}
        
        # Linear regression for each weather variable individually
        print(f"\nIndividual variable regression results for {target}:")
        
        reg_results = {}
        for var in weather_variables:
            try:
                # Skip if the variable has missing values
                if merged_df[var].isna().any() or merged_df[target].isna().any():
                    clean_df = merged_df[[var, target]].dropna()
                else:
                    clean_df = merged_df[[var, target]]
                    
                if len(clean_df) < 10:
                    print(f"  {var}: Not enough data points")
                    reg_results[var] = {'r2': None, 'coef': None, 'rmse': None}
                    continue
                    
                X = clean_df[var].values.reshape(-1, 1)
                y = clean_df[target].values
                
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                coef = model.coef_[0]
                
                print(f"  {var}: R² = {r2:.4f}, Coefficient = {coef:.4f}, RMSE = {rmse:.4f}")
                reg_results[var] = {'r2': r2, 'coef': coef, 'rmse': rmse}
                
                # Create scatter plot with regression line only for selected pollutants
                if detailed_analysis:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(merged_df[var], merged_df[target], alpha=0.4)
                    
                    # Add regression line
                    x_range = np.linspace(merged_df[var].min(), merged_df[var].max(), 100)
                    y_range = model.predict(x_range.reshape(-1, 1))
                    plt.plot(x_range, y_range, 'r-', linewidth=2)
                    
                    plt.xlabel(var)
                    plt.ylabel(target)
                    plt.title(f'Relationship between {var} and {target}\nR² = {r2:.4f}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{target}_{var}_regression.png"))
                    plt.close()
            except Exception as e:
                print(f"  Error with {var}: {e}")
                reg_results[var] = {'r2': None, 'coef': None, 'rmse': None, 'error': str(e)}
            
        # Store regression results
        target_results['regression'] = reg_results
        
        # Multiple linear regression with all weather variables
        try:
            # Prepare data for multiple regression
            X = merged_df[weather_variables].dropna()
            y = merged_df.loc[X.index, target]
            
            if len(X) < 10:
                print(f"Not enough data points for multiple regression for {target}")
                target_results['multiple_regression'] = {'r2': None, 'rmse': None, 'coefs': None}
            else:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                # Create coefficients DataFrame
                coefs = pd.DataFrame({
                    'Variable': weather_variables,
                    'Coefficient': model.coef_
                })
                coefs = coefs.sort_values(by='Coefficient', key=abs, ascending=False)
                
                print(f"\nMultiple Linear Regression for {target}:")
                print(f"R² = {r2:.4f}, RMSE = {rmse:.4f}")
                print("Variable importance:")
                print(coefs)
                
                target_results['multiple_regression'] = {
                    'r2': r2,
                    'rmse': rmse,
                    'coefs': coefs.to_dict('records')
                }
                
                # Create bar plot for variable importance only for selected pollutants
                if detailed_analysis:
                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(coefs['Variable'], coefs['Coefficient'])
                    
                    # Color positive and negative bars differently
                    for i, bar in enumerate(bars):
                        if coefs['Coefficient'].iloc[i] < 0:
                            bar.set_color('red')
                        else:
                            bar.set_color('green')
                            
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Coefficient')
                    plt.title(f'Weather Variable Importance for {target}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{target}_variable_importance.png"))
                    plt.close()
                
        except Exception as e:
            print(f"Error in multiple regression for {target}: {e}")
            target_results['multiple_regression'] = {'error': str(e)}
        
        # Store results for this target
        results[target] = target_results
    
    # Create summary table comparing R² values across pollutants
    summary = []
    for target in available_targets:
        if target in results and 'multiple_regression' in results[target]:
            r2 = results[target]['multiple_regression'].get('r2')
            if r2 is not None:
                summary.append({'Target': target, 'R²': r2})
    
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('R²', ascending=False)
        
        print("\nSummary of Multiple Regression R² Values:")
        print(summary_df)
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(summary_df['Target'], summary_df['R²'])
        plt.xlabel('R² Value')
        plt.title('Weather Variables - Pollutant Relationship Strength')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(1.0, summary_df['R²'].max() * 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pollutant_r2_comparison.png"))
        plt.close()
    
    return results
