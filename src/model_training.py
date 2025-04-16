#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voter Turnout Predictor: Model Training

This script trains and evaluates machine learning models to predict voter turnout
percentages at the county level.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

# For modeling
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Try to import XGBoost, but continue with a warning if it's not available
try:
    import xgboost as xgb

    has_xgboost = True
except ImportError:
    print("Warning: XGBoost not installed. Skipping XGBoost model.")
    has_xgboost = False

# Set plot styles
plt.style.use('seaborn-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = [12, 8]

# Set random seed for reproducibility
np.random.seed(42)


def load_data():
    """Load the processed dataset."""
    # Load the processed dataset
    data_path = 'data/processed/voter_turnout_features.csv'

    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        print("Trying fallback path...")

        # Try to load the merged dataset as fallback
        fallback_path = 'data/processed/voter_turnout_dataset.csv'
        if os.path.exists(fallback_path):
            print(f"Loading fallback data from {fallback_path}")
            df = pd.read_csv(fallback_path)
        else:
            raise FileNotFoundError("No suitable dataset found.")
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def prepare_data(df, target_col='turnout_percentage'):
    """Prepare features and target for modeling."""
    # Identify features and target
    y = df[target_col]

    # Exclude non-feature columns
    exclude_cols = ['fips', 'county_name', 'state', 'county', target_col]
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]

    print(f"Selected {len(feature_cols)} features for model training")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, feature_cols


def train_models(X_train, y_train):
    """Train several regression models."""
    # Define models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    # Add XGBoost if available
    if has_xgboost:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)

    # Dictionary to store trained models and their performance
    results = {}
    trained_models = {}

    # Train each model and evaluate with cross-validation
    for name, model in models.items():
        print(f"Training {name}...")

        # Train the model
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
        )
        rmse_scores = np.sqrt(-cv_scores)

        # Store CV results
        results[name] = {
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std()
        }

        print(f"  Cross-validation RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")

    return trained_models, results


def evaluate_models(trained_models, results, X_test, y_test):
    """Evaluate model performance on the test set."""
    # Evaluate each model
    for name, model in trained_models.items():
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store metrics
        results[name]['test_rmse'] = rmse
        results[name]['test_mae'] = mae
        results[name]['test_r2'] = r2

        print(f"{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

    # Create a DataFrame of results for easy comparison
    results_df = pd.DataFrame([
        {
            'Model': name,
            'CV RMSE': results[name]['cv_rmse_mean'],
            'Test RMSE': results[name]['test_rmse'],
            'Test MAE': results[name]['test_mae'],
            'Test R²': results[name]['test_r2']
        }
        for name in results.keys()
    ])

    # Sort by test RMSE (lower is better)
    results_df = results_df.sort_values('Test RMSE')

    return results_df


def visualize_model_performance(results_df):
    """Visualize the performance of different models."""
    # Create output directory for figures
    os.makedirs('reports/figures', exist_ok=True)

    # Plot RMSE (lower is better)
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    ax = sns.barplot(x='Model', y='Test RMSE', data=results_df)
    plt.title('Test RMSE by Model (Lower is Better)', fontsize=14)
    plt.xticks(rotation=45, ha='right')

    # Add value labels
    for i, v in enumerate(results_df['Test RMSE']):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center')

    # Plot R² (higher is better)
    plt.subplot(2, 1, 2)
    ax = sns.barplot(x='Model', y='Test R²', data=results_df)
    plt.title('Test R² by Model (Higher is Better)', fontsize=14)
    plt.xticks(rotation=45, ha='right')

    # Add value labels
    for i, v in enumerate(results_df['Test R²']):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig('reports/figures/model_performance_comparison.png')
    print("Saved model performance visualization to 'reports/figures/model_performance_comparison.png'")
    plt.close()


def tune_best_model(results_df, X_train, y_train):
    """Tune the hyperparameters of the best performing model."""
    # Identify the best performing model
    best_model_name = results_df.iloc[0]['Model']
    print(f"Best performing model: {best_model_name}")

    # Hyperparameter tuning depends on the model type
    if best_model_name == 'XGBoost' and has_xgboost:
        print("Tuning XGBoost hyperparameters...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        # Create model to tune
        model = xgb.XGBRegressor(random_state=42)

    elif best_model_name == 'Random Forest':
        print("Tuning Random Forest hyperparameters...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Create model to tune
        model = RandomForestRegressor(random_state=42)

    elif best_model_name == 'Gradient Boosting':
        print("Tuning Gradient Boosting hyperparameters...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10]
        }

        # Create model to tune
        model = GradientBoostingRegressor(random_state=42)

    elif 'Ridge' in best_model_name:
        print("Tuning Ridge Regression hyperparameters...")

        # Define parameter grid
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }

        # Create model to tune
        model = Ridge(random_state=42)

    else:
        print(f"No specific tuning defined for {best_model_name}, using default model")
        return None

    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all available cores
        verbose=1
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Output best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {np.sqrt(-grid_search.best_score_):.4f} RMSE")

    return best_model


def evaluate_tuned_model(best_model, X_test, y_test):
    """Evaluate the tuned model."""
    if best_model is None:
        print("No tuned model to evaluate.")
        return None

    # Evaluate the tuned model on the test set
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Tuned model performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

    return y_pred, rmse, mae, r2


def analyze_feature_importance(best_model, feature_cols):
    """Analyze and visualize feature importance."""
    if best_model is None:
        return

    # Create output directory for figures
    os.makedirs('reports/figures', exist_ok=True)

    # Check if model provides feature importance
    if hasattr(best_model, 'feature_importances_'):
        # Get feature importance
        importances = best_model.feature_importances_

        # Create a DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        })

        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        # Display the top features
        print("Top 15 most important features:")
        print(feature_importance.head(15))

        # Visualize feature importance
        plt.figure(figsize=(12, 8))

        # Plot top 15 features
        top_features = feature_importance.head(15)

        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Feature Importance', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()

        plt.savefig('reports/figures/feature_importance.png')
        print("Saved feature importance visualization to 'reports/figures/feature_importance.png'")
        plt.close()

        return feature_importance

    elif hasattr(best_model, 'coef_'):
        # For linear models
        coefficients = best_model.coef_

        # Create a DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': coefficients
        })

        # Sort by absolute coefficient value
        feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
        feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

        # Display the top features
        print("Top 15 features by coefficient magnitude:")
        print(feature_importance[['Feature', 'Coefficient']].head(15))

        # Visualize coefficients
        plt.figure(figsize=(12, 8))

        # Plot top 15 features
        top_features = feature_importance.head(15)

        sns.barplot(x='Coefficient', y='Feature', data=top_features)
        plt.title('Feature Coefficients', fontsize=16)
        plt.xlabel('Coefficient', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()

        plt.savefig('reports/figures/feature_coefficients.png')
        print("Saved feature coefficients visualization to 'reports/figures/feature_coefficients.png'")
        plt.close()

        return feature_importance

    else:
        print("This model doesn't provide feature importance information directly.")
        return None


def visualize_predictions(y_test, y_pred):
    """Visualize actual vs predicted values and residuals."""
    if y_pred is None:
        return

    # Create output directory for figures
    os.makedirs('reports/figures', exist_ok=True)

    # Visualize actual vs predicted values
    plt.figure(figsize=(12, 10))

    # Scatter plot of actual vs. predicted
    plt.subplot(2, 1, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs. Predicted Voter Turnout', fontsize=16)
    plt.xlabel('Actual Turnout (%)', fontsize=14)
    plt.ylabel('Predicted Turnout (%)', fontsize=14)

    # Residual plot
    plt.subplot(2, 1, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted Values', fontsize=16)
    plt.xlabel('Predicted Turnout (%)', fontsize=14)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=14)

    plt.tight_layout()
    plt.savefig('reports/figures/prediction_analysis.png')
    print("Saved prediction analysis to 'reports/figures/prediction_analysis.png'")
    plt.close()

    # Distribution of residuals
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Residuals', fontsize=16)
    plt.xlabel('Residual Value (Actual - Predicted)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.savefig('reports/figures/residual_distribution.png')
    print("Saved residual distribution to 'reports/figures/residual_distribution.png'")
    plt.close()

    # Print residual statistics
    print(f"Mean residual: {residuals.mean():.4f}")
    print(f"Residual standard deviation: {residuals.std():.4f}")
    print(f"Residual min: {residuals.min():.4f}")
    print(f"Residual max: {residuals.max():.4f}")


def save_model(best_model, feature_cols, rmse=None, mae=None, r2=None):
    """Save the best model and related information."""
    if best_model is None:
        return

    # Create directories for saving the model if they don't exist
    os.makedirs('models', exist_ok=True)

    # Save the best model
    model_filename = "models/voter_turnout_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Model saved to {model_filename}")

    # Save feature names for reference
    feature_filename = "models/feature_names.pkl"
    with open(feature_filename, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Feature names saved to {feature_filename}")

    # Save model performance metrics if available
    if rmse is not None and mae is not None and r2 is not None:
        performance = {
            'model_type': type(best_model).__name__,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_count': len(feature_cols)
        }

        performance_filename = "models/model_performance.pkl"
        with open(performance_filename, 'wb') as f:
            pickle.dump(performance, f)
        print(f"Performance metrics saved to {performance_filename}")


def generate_future_predictions(df, best_model, feature_cols):
    """Generate predictions for future elections."""
    if best_model is None or 'year' not in df.columns:
        return

    # Get the most recent data as a starting point
    most_recent_year = df['year'].max()
    recent_data = df[df['year'] == most_recent_year].copy()
    print(f"Using {len(recent_data)} counties from {most_recent_year} as baseline for 2024 predictions")

    # Update the year to 2024
    future_data = recent_data.copy()
    future_data['year'] = 2024

    # Make some assumptions about how demographics might change
    if 'higher_education_percentage' in future_data.columns:
        future_data['higher_education_percentage'] *= 1.05  # 5% increase

    if 'median_household_income' in future_data.columns:
        future_data['median_household_income'] *= 1.08  # 8% increase

    if 'unemployment_rate' in future_data.columns:
        future_data['unemployment_rate'] *= 0.9  # 10% decrease

    # Assume higher ad spending in a presidential election year
    if 'ad_spend_per_capita' in future_data.columns:
        future_data['ad_spend_per_capita'] *= 1.2  # 20% increase

    # Update any year-specific features that might have been created
    if 'election_year' in future_data.columns:
        future_data['election_year'] = '2024'

    # Ensure we only use features the model knows about
    future_feature_cols = [col for col in feature_cols if col in future_data.columns]
    missing_features = [col for col in feature_cols if col not in future_data.columns]

    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features for prediction. Model may not be accurate.")
        print(f"Missing features: {missing_features[:5]}...")

    # Prepare features for prediction
    future_X = future_data[future_feature_cols]

    # Fill missing columns with zeros (not ideal but allows prediction)
    for col in feature_cols:
        if col not in future_X.columns:
            future_X[col] = 0

    # Reorder columns to match the model's expected order
    future_X = future_X[feature_cols]

    # Make predictions
    future_predictions = best_model.predict(future_X)

    # Add predictions to the future dataset
    future_data['predicted_turnout'] = future_predictions

    # View summary statistics of predictions
    print("\n2024 Turnout Predictions:")
    print(f"Average predicted turnout: {future_predictions.mean():.2f}%")
    print(f"Minimum predicted turnout: {future_predictions.min():.2f}%")
    print(f"Maximum predicted turnout: {future_predictions.max():.2f}%")

    # Save predictions
    prediction_cols = ['fips', 'state_abbr', 'county_name', 'predicted_turnout']
    prediction_cols = [col for col in prediction_cols if col in future_data.columns]

    predictions_df = future_data[prediction_cols]
    predictions_df = predictions_df.sort_values('predicted_turnout')

    # Save to CSV
    os.makedirs('data/predictions', exist_ok=True)
    predictions_df.to_csv('data/predictions/turnout_predictions_2024.csv', index=False)
    print(f"Predictions saved to 'data/predictions/turnout_predictions_2024.csv'")

    # Display counties with lowest predicted turnout
    print("\nCounties with lowest predicted turnout:")
    print(predictions_df.head(10))


def create_model_report(results_df, best_model, feature_importance=None, rmse=None, mae=None, r2=None):
    """Create a markdown report with model insights."""
    if best_model is None:
        return

    # Create the reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)

    # Start building the report
    report = f"""# Voter Turnout Prediction Model: Insights and Interpretation

## Model Performance
- **Best Model:** {type(best_model).__name__}
"""

    if rmse is not None and mae is not None and r2 is not None:
        report += f"""- **Root Mean Square Error (RMSE):** {rmse:.2f}%
- **Mean Absolute Error (MAE):** {mae:.2f}%
- **R² Score:** {r2:.2f}

"""

    # Add model comparison
    report += """## Model Comparison

| Model | Test RMSE | Test MAE | Test R² |
| ----- | --------- | -------- | ------- |
"""

    for _, row in results_df.iterrows():
        report += f"| {row['Model']} | {row['Test RMSE']:.2f} | {row['Test MAE']:.2f} | {row['Test R²']:.2f} |\n"

    report += "\n## Key Predictors of Voter Turnout\n\n"

    # Add feature importance if available
    if feature_importance is not None:
        top_n = 10

        if 'Importance' in feature_importance.columns:
            report += "### Top 10 Most Influential Features\n\n"
            report += "| Feature | Importance |\n| ------ | ---------- |\n"

            for _, row in feature_importance.head(top_n).iterrows():
                report += f"| {row['Feature']} | {row['Importance']:.4f} |\n"

        elif 'Coefficient' in feature_importance.columns:
            report += "### Top 10 Features by Coefficient Magnitude\n\n"
            report += "| Feature | Coefficient |\n| ------ | ---------- |\n"

            for _, row in feature_importance.head(top_n).iterrows():
                report += f"| {row['Feature']} | {row['Coefficient']:.4f} |\n"

    # Add actionable insights
    report += """
## Actionable Insights

Based on the model predictions and feature importance:

1. **Education Impact:** Higher education levels are strongly associated with increased voter turnout. Civic education initiatives in areas with lower educational attainment might be effective.

2. **Economic Factors:** Household income shows significant correlation with turnout. Economic development in lower-income counties could indirectly boost civic participation.

3. **Campaign Resource Allocation:** Ad spending effectiveness varies by county demographics. The model can help target resources to areas where they'll have the greatest impact on turnout.

4. **Targeted Interventions:** Counties with the lowest predicted turnout should be prioritized for get-out-the-vote campaigns and improved voting access.

5. **Demographic Trends:** Understanding how changing demographics will affect future turnout can help with long-term planning and resource allocation.

## Next Steps

1. **Dashboard Development:** Create an interactive dashboard to visualize predictions and identify turnout hotspots.

2. **Regular Model Updates:** Retrain the model as new election data becomes available to maintain accuracy.

3. **More Granular Analysis:** Consider analysis at more detailed levels (precincts or census tracts) where data is available.

4. **Additional Features:** Incorporate additional data sources like polling place proximity, mail voting policies, and historical weather on election days.

5. **Causal Analysis:** Conduct experiments to move from correlation to causation, testing which interventions actually increase turnout.
"""

    # Save the report
    with open('reports/model_report.md', 'w') as f:
        f.write(report)

    print("Model report saved to 'reports/model_report.md'")


def main():
    """Main function to run the model training pipeline."""
    print("Voter Turnout Predictor: Model Training")
    print("======================================")

    try:
        # 1. Load data
        print("\n1. Loading data...")
        df = load_data()

        # 2. Prepare data
        print("\n2. Preparing data for modeling...")
        X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)

        # 3. Train models
        print("\n3. Training machine learning models...")
        trained_models, results = train_models(X_train, y_train)

        # 4. Evaluate models
        print("\n4. Evaluating model performance...")
        results_df = evaluate_models(trained_models, results, X_test, y_test)

        # 5. Visualize model performance
        print("\n5. Visualizing model performance...")
        visualize_model_performance(results_df)

        # 6. Tune best model
        print("\n6. Tuning the best performing model...")
        best_model = tune_best_model(results_df, X_train, y_train)

        # 7. Evaluate tuned model
        print("\n7. Evaluating the tuned model...")
        if best_model is not None:
            y_pred, rmse, mae, r2 = evaluate_tuned_model(best_model, X_test, y_test)
        else:
            print("Using the best untuned model instead...")
            best_model_name = results_df.iloc[0]['Model']
            best_model = trained_models[best_model_name]
            y_pred = best_model.predict(X_test)
            rmse = results_df.iloc[0]['Test RMSE']
            mae = results_df.iloc[0]['Test MAE']
            r2 = results_df.iloc[0]['Test R²']

        # 8. Analyze feature importance
        print("\n8. Analyzing feature importance...")
        feature_importance = analyze_feature_importance(best_model, feature_cols)

        # 9. Visualize predictions
        print("\n9. Visualizing predictions...")
        visualize_predictions(y_test, y_pred)

        # 10. Save the model
        print("\n10. Saving the final model...")
        save_model(best_model, feature_cols, rmse, mae, r2)

        # 11. Generate future predictions
        print("\n11. Generating predictions for future elections...")
        generate_future_predictions(df, best_model, feature_cols)

        # 12. Create model report
        print("\n12. Creating model report...")
        create_model_report(results_df, best_model, feature_importance, rmse, mae, r2)

        print("\nModel training complete! 🎉")
        print(f"Best model: {type(best_model).__name__}")
        print(f"Model performance: RMSE = {rmse:.2f}, R² = {r2:.2f}")
        print("Model saved to 'models/voter_turnout_model.joblib'")
        print("Run the dashboard creator to visualize the predictions!")

    except Exception as e:
        print(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()