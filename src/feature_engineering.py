#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voter Turnout Predictor: Feature Engineering

This script performs feature engineering on the voter turnout dataset to prepare it
for machine learning models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set plot styles
try:
    plt.style.use('seaborn-whitegrid')  # For older versions of matplotlib
except:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')  # For newer versions of matplotlib
    except:
        print("Seaborn style not found, using default style")
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = [12, 8]


def load_data():
    """Load the merged dataset."""
    # Load the merged dataset
    data_path = 'data/processed/voter_turnout_dataset.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}. Run the data merger script first.")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    return df


def explore_data(df):
    """Explore data characteristics."""
    print("\nExploring data characteristics...")

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])

    # Basic statistics
    print("\nSummary statistics for key columns:")
    key_cols = ['turnout_percentage', 'median_household_income', 'unemployment_rate']
    key_cols = [col for col in key_cols if col in df.columns]

    if key_cols:
        print(df[key_cols].describe())

    # Create output directory for figures
    os.makedirs('reports/figures', exist_ok=True)

    # Visualize the distribution of the target variable
    if 'turnout_percentage' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['turnout_percentage'], kde=True, bins=30)
        plt.title('Distribution of Voter Turnout Percentage', fontsize=16)
        plt.xlabel('Turnout Percentage', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.savefig('reports/figures/turnout_distribution.png')
        plt.close()
        print("Saved turnout distribution plot to 'reports/figures/turnout_distribution.png'")

    # Visualize turnout by year
    if 'year' in df.columns and 'turnout_percentage' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='year', y='turnout_percentage', data=df)
        plt.title('Voter Turnout by Election Year', fontsize=16)
        plt.xlabel('Election Year', fontsize=14)
        plt.ylabel('Turnout Percentage', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig('reports/figures/turnout_by_year.png')
        plt.close()
        print("Saved turnout by year plot to 'reports/figures/turnout_by_year.png'")


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\nHandling missing values...")

    # Check if there are any missing values
    if df.isnull().sum().sum() == 0:
        print("No missing values found.")
        return df

    # For numerical columns, fill with median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"  - Filled {col} missing values with median ({median_value:.2f})")

    # For categorical columns, fill with mode
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            print(f"  - Filled {col} missing values with mode ({mode_value})")

    return df


def handle_outliers(df, columns=None):
    """Handle outliers in specified columns using IQR method."""
    print("\nHandling outliers...")

    # If no columns specified, use these default columns if they exist
    if columns is None:
        default_cols = ['median_household_income', 'unemployment_rate', 'turnout_percentage']
        columns = [col for col in default_cols if col in df.columns]

    # Create output directory for figures
    os.makedirs('reports/figures', exist_ok=True)

    # Process each column
    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in the dataset. Skipping.")
            continue

        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"  - Column '{col}' has {len(outliers)} outliers ({len(outliers) / len(df) * 100:.2f}%)")

        # Visualize before capping
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.boxplot(x=df[col])
        plt.title(f'Before Handling Outliers: {col}')

        # Cap outliers
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        # Visualize after capping
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'After Handling Outliers: {col}')

        plt.tight_layout()
        plt.savefig(f'reports/figures/outliers_{col}.png')
        plt.close()
        print(f"  - Saved outlier handling visualization to 'reports/figures/outliers_{col}.png'")

    return df


def create_features(df):
    """Create new features from existing ones."""
    print("\nCreating new features...")

    # Store original number of columns
    original_cols = df.shape[1]

    # 1. Interaction features
    if 'median_household_income' in df.columns and 'higher_education_percentage' in df.columns:
        df['income_education_interaction'] = df['median_household_income'] * df['higher_education_percentage'] / 1000
        print("  - Created income_education_interaction")

    # 2. Squared terms for non-linear relationships
    if 'unemployment_rate' in df.columns:
        df['unemployment_rate_squared'] = df['unemployment_rate'] ** 2
        print("  - Created unemployment_rate_squared")

    if 'higher_education_percentage' in df.columns:
        df['higher_education_squared'] = df['higher_education_percentage'] ** 2
        print("  - Created higher_education_squared")

    # 3. Log transform for skewed distributions
    if 'total_population' in df.columns:
        df['log_population'] = np.log1p(df['total_population'])
        print("  - Created log_population")

    # 4. Ad spending features
    if 'republican_ad_spend' in df.columns and 'democrat_ad_spend' in df.columns:
        # Ratio of Republican to Democrat spending
        epsilon = 1  # Small constant to avoid division by zero
        df['rep_dem_spend_ratio'] = df['republican_ad_spend'] / (df['democrat_ad_spend'] + epsilon)

        # Total party spend
        df['total_party_spend'] = df['republican_ad_spend'] + df['democrat_ad_spend']

        # Spending difference
        df['spend_difference'] = abs(df['republican_ad_spend'] - df['democrat_ad_spend'])

        print("  - Created ad spending features")

    # 5. Previous election features (for years after the first election)
    if 'year' in df.columns and len(df['year'].unique()) > 1 and 'turnout_percentage' in df.columns:
        if 'fips' in df.columns:
            # Sort by county (FIPS) and year
            df = df.sort_values(['fips', 'year'])

            # Create a column for the previous turnout (shifted within each county group)
            df['prev_turnout'] = df.groupby('fips')['turnout_percentage'].shift(1)

            # Fill missing values (counties with only one election) with the overall average
            df['prev_turnout'] = df['prev_turnout'].fillna(df['turnout_percentage'].mean())

            # Create turnout change feature
            df['turnout_change'] = df['turnout_percentage'] - df['prev_turnout']

            print("  - Created previous_turnout and turnout_change features")

    # 6. Urban/rural classification based on population
    if 'total_population' in df.columns:
        population_bins = [0, 50000, 100000, 500000, float('inf')]
        population_labels = ['rural', 'small_urban', 'medium_urban', 'large_urban']
        df['urban_rural_category'] = pd.cut(df['total_population'], bins=population_bins, labels=population_labels)
        print("  - Created urban_rural_category")

    # 7. Election year as categorical
    if 'year' in df.columns:
        df['election_year'] = df['year'].astype(str).astype('category')
        print("  - Created election_year categorical feature")

    # 8. Swing state interaction
    if 'is_swing_state' in df.columns and 'ad_spend_per_capita' in df.columns:
        df['swing_state_ad_spend'] = df['is_swing_state'].astype(int) * df['ad_spend_per_capita']
        print("  - Created swing_state_ad_spend interaction")

    # Report how many features were added
    print(f"\nAdded {df.shape[1] - original_cols} new features.")

    return df


def encode_categorical_features(df):
    """Encode categorical variables using one-hot encoding."""
    print("\nEncoding categorical features...")

    # Identify categorical columns (object dtype or categorical)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Exclude columns we don't want to encode (like names and identifiers)
    exclude_cols = ['county_name', 'state', 'county', 'fips', 'state_abbr']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols and col in df.columns]

    print(f"Categorical columns to encode: {categorical_cols}")

    # One-hot encode each categorical column
    for col in categorical_cols:
        if col in df.columns:
            # Get dummies (one-hot encoding)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)

            # Concatenate with original dataframe
            df = pd.concat([df, dummies], axis=1)

            print(f"  - Encoded '{col}' into {dummies.shape[1]} dummy variables")

    return df


def scale_numerical_features(df, target_col='turnout_percentage'):
    """Scale numerical features using StandardScaler."""
    print("\nScaling numerical features...")

    # Identify numerical columns for scaling (exclude target and IDs)
    exclude_cols = ['year', 'fips', 'total_votes_cast', 'total_votes_reported',
                    'total_population', target_col]
    exclude_cols = [col for col in exclude_cols if col in df.columns]

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

    print(f"Numerical columns to scale: {len(numerical_cols)} columns")

    # Create a copy for scaled data
    df_scaled = df.copy()

    # Initialize the scaler
    scaler = StandardScaler()

    # Scale the numerical features
    if numerical_cols:
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Visualize before and after scaling for a few selected columns
        sample_cols = numerical_cols[:min(3, len(numerical_cols))]

        for col in sample_cols:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True)
            plt.title(f'Before Scaling: {col}')

            plt.subplot(1, 2, 2)
            sns.histplot(df_scaled[col], kde=True)
            plt.title(f'After Scaling: {col}')

            plt.tight_layout()
            plt.savefig(f'reports/figures/scaling_{col}.png')
            plt.close()
            print(f"  - Saved scaling visualization for '{col}' to 'reports/figures/scaling_{col}.png'")

    return df, df_scaled


def analyze_correlations(df, target_col='turnout_percentage'):
    """Analyze correlations between features and the target variable."""
    print("\nAnalyzing correlations with turnout...")

    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in the dataset. Skipping correlation analysis.")
        return

    # Calculate correlations with target
    correlations = df.corr()[target_col].sort_values(ascending=False)

    # Display top positive and negative correlations
    print("\nTop positive correlations with turnout:")
    print(correlations.head(10))

    print("\nTop negative correlations with turnout:")
    print(correlations.tail(10))

    # Create correlation heatmap for top features
    # Get top 15 features by absolute correlation
    top_features = correlations.drop(target_col).abs().sort_values(ascending=False).head(15).index

    # Create correlation matrix
    corr_matrix = df[list(top_features) + [target_col]].corr()

    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Between Top Features and Voter Turnout', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png')
    plt.close()
    print("Saved correlation heatmap to 'reports/figures/correlation_heatmap.png'")

    # Create scatter plots for top correlated features
    top_corr_features = correlations.drop(target_col).abs().sort_values(ascending=False).head(4).index

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, feature in enumerate(top_corr_features):
        sns.scatterplot(ax=axes[i], x=feature, y=target_col, data=df, alpha=0.6)
        axes[i].set_title(f'{feature} vs. {target_col}', fontsize=14)
        axes[i].set_xlabel(feature, fontsize=12)
        axes[i].set_ylabel(target_col, fontsize=12)

        # Add regression line
        sns.regplot(x=feature, y=target_col, data=df, ax=axes[i], scatter=False, color='red')

    plt.tight_layout()
    plt.savefig('reports/figures/top_correlations.png')
    plt.close()
    print("Saved top correlation scatter plots to 'reports/figures/top_correlations.png'")


def save_processed_data(df, df_scaled=None):
    """Save the processed datasets."""
    print("\nSaving processed data...")

    # Create output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)

    # Save the processed data
    df.to_csv('data/processed/voter_turnout_features.csv', index=False)
    print("Saved processed features to 'data/processed/voter_turnout_features.csv'")

    # Save scaled data if available
    if df_scaled is not None:
        df_scaled.to_csv('data/processed/voter_turnout_features_scaled.csv', index=False)
        print("Saved scaled features to 'data/processed/voter_turnout_features_scaled.csv'")


def main():
    """Main function to perform feature engineering."""
    print("Voter Turnout Predictor: Feature Engineering")
    print("===========================================")

    try:
        # 1. Load data
        print("\n1. Loading data...")
        df = load_data()

        # 2. Explore data
        print("\n2. Exploring data characteristics...")
        explore_data(df)

        # 3. Handle missing values
        print("\n3. Handling missing values...")
        df = handle_missing_values(df)

        # 4. Handle outliers
        print("\n4. Handling outliers...")
        df = handle_outliers(df)

        # 5. Create features
        print("\n5. Creating new features...")
        df = create_features(df)

        # 6. Encode categorical features
        print("\n6. Encoding categorical features...")
        df = encode_categorical_features(df)

        # 7. Scale numerical features
        print("\n7. Scaling numerical features...")
        df, df_scaled = scale_numerical_features(df)

        # 8. Analyze correlations
        print("\n8. Analyzing feature correlations...")
        analyze_correlations(df)

        # 9. Save processed data
        print("\n9. Saving processed data...")
        save_processed_data(df, df_scaled)

        print("\nFeature engineering complete! 🎉")
        print("Processed data saved to 'data/processed/voter_turnout_features.csv'")
        print("You can now proceed to model training.")

    except Exception as e:
        print(f"Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()