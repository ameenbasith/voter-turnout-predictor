#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script directly processes and merges the available data files
to create a dataset ready for modeling.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set up directories
data_dir = 'data'
if not os.path.exists(data_dir):
    data_dir = 'src/data'

raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_dir, exist_ok=True)

# Set up output directories for reports and figures
reports_dir = 'reports'
figures_dir = os.path.join(reports_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)


def load_census_data():
    """Load and process census data."""
    print("Loading census data...")

    # Try to load the combined census data first
    census_file = os.path.join(raw_dir, 'census_county_data_all_years.csv')

    if os.path.exists(census_file):
        census_df = pd.read_csv(census_file)
        print(f"Loaded combined census data with {len(census_df)} records.")
        return census_df

    # If not available, try to load individual year files
    census_dfs = []
    for year in [2012, 2016, 2020]:
        year_file = os.path.join(raw_dir, f'census_county_data_{year}.csv')
        if os.path.exists(year_file):
            year_df = pd.read_csv(year_file)
            census_dfs.append(year_df)
            print(f"Loaded census data for {year} with {len(year_df)} records.")

    if census_dfs:
        combined_df = pd.concat(census_dfs, ignore_index=True)
        print(f"Combined census data with {len(combined_df)} total records.")
        return combined_df

    raise FileNotFoundError("Could not find any census data files.")


def load_election_data():
    """Load and process election data."""
    print("Loading election data...")

    # Try to load the election data
    election_file = os.path.join(raw_dir, 'countypres_2000-2020.csv')

    if not os.path.exists(election_file):
        raise FileNotFoundError(f"Election data file not found at {election_file}")

    election_df = pd.read_csv(election_file)
    print(f"Loaded election data with {len(election_df)} records.")

    # Filter to include only 2012, 2016, and 2020 elections
    election_df = election_df[election_df['year'].isin([2012, 2016, 2020])]
    print(f"Filtered to {len(election_df)} records for 2012, 2016, and 2020 elections.")

    # Process election data to get county-level turnout
    # Filter to TOTAL mode if it exists
    if 'mode' in election_df.columns:
        election_df = election_df[election_df['mode'] == 'TOTAL']
        print(f"Filtered to {len(election_df)} records for TOTAL mode.")

    # Group by county and year to get total votes
    group_cols = ['year', 'state', 'county_fips', 'county_name']

    # Check if we need to add the state_po column
    if 'state_po' in election_df.columns:
        group_cols.append('state_po')

    # Create aggregated data
    if 'candidatevotes' in election_df.columns and 'totalvotes' in election_df.columns:
        county_votes = election_df.groupby(group_cols).agg({
            'candidatevotes': 'sum',  # Total votes for all candidates
            'totalvotes': 'first'  # Total votes should be the same for all candidates
        }).reset_index()

        county_votes = county_votes.rename(columns={
            'candidatevotes': 'total_votes_cast',
            'totalvotes': 'total_votes_reported'
        })
    else:
        # Try to find any column with 'vote' in it
        vote_cols = [col for col in election_df.columns if 'vote' in col.lower()]
        if vote_cols:
            county_votes = election_df.groupby(group_cols)[vote_cols].sum().reset_index()
            # Rename first vote column to total_votes_cast
            county_votes = county_votes.rename(columns={vote_cols[0]: 'total_votes_cast'})
            county_votes['total_votes_reported'] = county_votes['total_votes_cast']
        else:
            raise ValueError("Could not identify vote count columns in the election data.")

    # Rename state_po to state_abbr if needed
    if 'state_po' in county_votes.columns:
        county_votes = county_votes.rename(columns={'state_po': 'state_abbr'})

    # Ensure FIPS codes are properly formatted as strings with leading zeros
    if 'county_fips' in county_votes.columns:
        county_votes['county_fips'] = county_votes['county_fips'].astype(str).str.zfill(5)

        # Extract state and county FIPS codes
        county_votes['state_fips'] = county_votes['county_fips'].str[:2]
        county_votes['county_fips_only'] = county_votes['county_fips'].str[2:]

    print(f"Created county-level voting data with {len(county_votes)} records.")
    return county_votes


def create_swing_state_data():
    """Create data about swing states for different election years."""
    swing_states = {
        2012: ['FL', 'OH', 'NC', 'VA', 'WI', 'CO', 'IA', 'NH', 'NV'],
        2016: ['FL', 'PA', 'OH', 'MI', 'NC', 'WI', 'AZ', 'CO', 'IA', 'NH', 'NV'],
        2020: ['FL', 'PA', 'MI', 'NC', 'WI', 'AZ', 'GA', 'MN', 'NH', 'NV']
    }

    # Create dataframe with swing state indicator
    swing_state_data = []

    for year, states in swing_states.items():
        for state in states:
            swing_state_data.append({
                'year': year,
                'state_abbr': state,
                'is_swing_state': True
            })

    swing_df = pd.DataFrame(swing_state_data)
    print(f"Created swing state data for {len(swing_state_data)} state-year combinations.")
    return swing_df


def simulate_ad_spending(merged_df):
    """Simulate campaign ad spending data."""
    print("Simulating campaign ad spending data...")

    # Make a copy of the input dataframe
    df = merged_df.copy()

    # Base spending per capita (in dollars)
    base_spending_per_capita = {
        2012: 2.5,
        2016: 3.0,
        2020: 3.5
    }

    # Swing state multiplier
    swing_state_multiplier = 5  # Spend 5x more in swing states

    # Add random variation to make it more realistic
    np.random.seed(42)  # For reproducibility
    df['ad_spend_multiplier'] = np.random.lognormal(mean=0, sigma=0.5, size=len(df))

    # Calculate ad spending
    def calculate_ad_spend(row):
        year = row['year']
        base_rate = base_spending_per_capita.get(year, 3.0)
        multiplier = swing_state_multiplier if row.get('is_swing_state', False) else 1
        random_factor = row['ad_spend_multiplier']

        # Calculate based on population
        population = row.get('total_population', 100000)  # Use default if not available
        if pd.isna(population) or population <= 0:
            population = 100000  # Default population

        # More spending in counties with higher education (targeted ads)
        education_factor = 1.0
        if 'higher_education_percentage' in row:
            education_factor = 1.0 + row['higher_education_percentage'] / 100

        # Base calculation
        ad_spend = population * base_rate * multiplier * random_factor * education_factor

        # Add some political party bias
        rep_spend = ad_spend * np.random.uniform(0.8, 1.2)
        dem_spend = ad_spend * np.random.uniform(0.8, 1.2)

        return {
            'total_ad_spend': ad_spend,
            'republican_ad_spend': rep_spend,
            'democrat_ad_spend': dem_spend
        }

    # Apply the calculation
    ad_spend_data = df.apply(calculate_ad_spend, axis=1, result_type='expand')

    # Add to dataframe
    df['total_ad_spend'] = ad_spend_data['total_ad_spend']
    df['republican_ad_spend'] = ad_spend_data['republican_ad_spend']
    df['democrat_ad_spend'] = ad_spend_data['democrat_ad_spend']

    # Round to nearest dollar
    df['total_ad_spend'] = df['total_ad_spend'].round(0)
    df['republican_ad_spend'] = df['republican_ad_spend'].round(0)
    df['democrat_ad_spend'] = df['democrat_ad_spend'].round(0)

    # Add ad spend per capita
    if 'total_population' in df.columns:
        df['ad_spend_per_capita'] = df['total_ad_spend'] / df['total_population']

    print("Added campaign ad spending data.")
    return df


def merge_datasets():
    """Merge all datasets into a single dataframe."""
    print("Merging datasets...")

    try:
        # Load data
        election_df = load_election_data()
        census_df = load_census_data()
        swing_df = create_swing_state_data()

        # Standardize FIPS codes
        if 'fips' in census_df.columns:
            census_df['county_fips'] = census_df['fips']

        if 'county_fips' in census_df.columns:
            census_df['county_fips'] = census_df['county_fips'].astype(str).str.zfill(5)

        # Merge election data with swing state data
        merged_df = pd.merge(
            election_df,
            swing_df,
            on=['year', 'state_abbr'],
            how='left'
        )

        # Fill missing swing state values with False
        merged_df['is_swing_state'] = merged_df['is_swing_state'].fillna(False)

        # Print schema information to help with debugging
        print("\nElection data columns:", election_df.columns.tolist())
        print("\nCensus data columns:", census_df.columns.tolist())

        # Choose the right merge key based on available columns
        if 'county_fips' in census_df.columns and 'county_fips' in merged_df.columns:
            # Merge on FIPS code and year
            print("\nMerging on county_fips and year...")
            final_df = pd.merge(
                merged_df,
                census_df,
                on=['county_fips', 'year'],
                how='left',
                suffixes=('', '_census')
            )
        elif 'fips' in census_df.columns and 'county_fips' in merged_df.columns:
            # Convert to same format
            print("\nMerging on fips/county_fips and year...")
            census_df['county_fips'] = census_df['fips']
            final_df = pd.merge(
                merged_df,
                census_df,
                on=['county_fips', 'year'],
                how='left',
                suffixes=('', '_census')
            )
        else:
            # Try merge on county name, state, and year
            print("\nMerging on county_name, state, and year...")
            # Make sure state columns match
            if 'state_abbr' in merged_df.columns and 'state' in census_df.columns:
                census_df['state_abbr'] = census_df['state']

            final_df = pd.merge(
                merged_df,
                census_df,
                on=['county_name', 'state', 'year'],
                how='left',
                suffixes=('', '_census')
            )

        print(f"Merged dataset has {len(final_df)} rows.")

        # Calculate voter turnout percentage if we have population data
        if 'total_population' in final_df.columns and 'total_votes_cast' in final_df.columns:
            # Voting Age Population (VAP) is typically about 75% of total population
            final_df['estimated_vap'] = final_df['total_population'] * 0.75

            # Calculate turnout
            final_df['turnout_percentage'] = (final_df['total_votes_cast'] / final_df['estimated_vap']) * 100

            # Cap turnout at 100%
            final_df['turnout_percentage'] = final_df['turnout_percentage'].clip(upper=100)

            print("Added voter turnout percentages.")

        # Add higher education percentage if the components are available
        education_cols = ['bachelors_degree', 'masters_degree', 'professional_degree', 'doctorate_degree']
        if all(col in final_df.columns for col in education_cols) and 'total_population' in final_df.columns:
            final_df['higher_education'] = final_df['bachelors_degree'] + \
                                           final_df['masters_degree'] + \
                                           final_df['professional_degree'] + \
                                           final_df['doctorate_degree']

            final_df['higher_education_percentage'] = (final_df['higher_education'] / final_df[
                'total_population']) * 100
            print("Added higher education percentage.")

        # Simulate ad spending data
        final_df = simulate_ad_spending(final_df)

        # Clean up duplicate columns
        duplicate_cols = [col for col in final_df.columns if col.endswith('_census')]
        final_df = final_df.drop(columns=duplicate_cols, errors='ignore')

        # Drop rows with missing turnout percentage
        if 'turnout_percentage' in final_df.columns:
            missing_turnout = final_df['turnout_percentage'].isna().sum()
            if missing_turnout > 0:
                print(f"Warning: {missing_turnout} rows have missing turnout percentages.")

                # For demonstration, we'll fill with median rather than dropping
                final_df['turnout_percentage'] = final_df['turnout_percentage'].fillna(
                    final_df['turnout_percentage'].median())

        return final_df

    except Exception as e:
        print(f"Error merging datasets: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_features(df):
    """Create features for machine learning."""
    print("\nCreating features...")

    # Store original number of columns
    original_cols = df.shape[1]

    # 1. Log transform population (often better for modeling)
    if 'total_population' in df.columns:
        df['log_population'] = np.log1p(df['total_population'])
        print("  - Created log_population")

    # 2. Interaction between education and income
    if 'higher_education_percentage' in df.columns and 'median_household_income' in df.columns:
        df['income_education_interaction'] = df['higher_education_percentage'] * df['median_household_income'] / 1000
        print("  - Created income_education_interaction")

    # 3. Swing state and ad spending interaction
    if 'is_swing_state' in df.columns and 'ad_spend_per_capita' in df.columns:
        df['swing_state_ad_spend'] = df['is_swing_state'].astype(int) * df['ad_spend_per_capita']
        print("  - Created swing_state_ad_spend interaction")

    # 4. Party spending ratio
    if 'republican_ad_spend' in df.columns and 'democrat_ad_spend' in df.columns:
        # Add small epsilon to avoid division by zero
        epsilon = 1.0
        df['rep_dem_ratio'] = df['republican_ad_spend'] / (df['democrat_ad_spend'] + epsilon)
        print("  - Created rep_dem_ratio")

    # 5. Previous election turnout (for years after 2012)
    if 'year' in df.columns and 'turnout_percentage' in df.columns and 'county_fips' in df.columns:
        # Sort by county and year
        df = df.sort_values(['county_fips', 'year'])

        # Create previous turnout column (shifted within each county)
        df['prev_turnout'] = df.groupby('county_fips')['turnout_percentage'].shift(1)

        # Fill missing values (for 2012, the earliest year) with median
        df['prev_turnout'] = df['prev_turnout'].fillna(df['turnout_percentage'].median())

        # Calculate turnout change
        df['turnout_change'] = df['turnout_percentage'] - df['prev_turnout']

        print("  - Created prev_turnout and turnout_change")

    # 6. County population categories (rural/urban)
    if 'total_population' in df.columns:
        # Define population bins and labels
        pop_bins = [0, 10000, 50000, 100000, 500000, float('inf')]
        pop_labels = ['very_rural', 'rural', 'suburban', 'urban', 'major_urban']

        # Create the categorical variable
        df['population_category'] = pd.cut(df['total_population'], bins=pop_bins, labels=pop_labels)
        print("  - Created population_category")

    # 7. Higher education brackets
    if 'higher_education_percentage' in df.columns:
        # Define education bins and labels
        edu_bins = [0, 10, 20, 30, 40, float('inf')]
        edu_labels = ['very_low', 'low', 'medium', 'high', 'very_high']

        # Create the categorical variable
        df['education_level'] = pd.cut(df['higher_education_percentage'], bins=edu_bins, labels=edu_labels)
        print("  - Created education_level")

    # 8. Election year as categorical
    if 'year' in df.columns:
        df['election_year'] = df['year'].astype(str)
        print("  - Created election_year")

    # Report how many features were added
    print(f"Added {df.shape[1] - original_cols} new features.")
    return df


def handle_outliers(df):
    """Handle outliers in key numerical columns."""
    print("\nHandling outliers...")

    # Define columns to check for outliers
    columns = ['turnout_percentage', 'median_household_income', 'unemployment_rate',
               'ad_spend_per_capita', 'higher_education_percentage']
    columns = [col for col in columns if col in df.columns]

    for col in columns:
        # Calculate quartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers_mask.sum()

        if outlier_count > 0:
            print(f"  - Capping {outlier_count} outliers in {col}")

            # Cap outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def encode_categorical_features(df):
    """One-hot encode categorical features."""
    print("\nEncoding categorical features...")

    # Identify categorical columns
    categorical_cols = []

    # Check known categorical columns
    potential_categorical = ['population_category', 'education_level', 'election_year']
    for col in potential_categorical:
        if col in df.columns:
            categorical_cols.append(col)

    # Add any other object or category columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col not in categorical_cols and col not in ['county_name', 'state', 'state_abbr']:
            categorical_cols.append(col)

    print(f"Found {len(categorical_cols)} categorical columns to encode")

    # Encode each categorical column
    for col in categorical_cols:
        # Get dummies (one-hot encoding)
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)

        # Add to dataframe
        df = pd.concat([df, dummies], axis=1)

        print(f"  - Encoded {col} into {dummies.shape[1]} dummy variables")

    return df


def scale_features(df, target_col='turnout_percentage'):
    """Scale numerical features."""
    print("\nScaling numerical features...")

    # Identify numerical columns (exclude target, IDs, etc.)
    exclude_cols = ['year', 'fips', 'county_fips', 'state_fips', 'county_fips_only',
                    'total_votes_cast', 'total_votes_reported', 'total_population',
                    target_col]
    exclude_cols = [col for col in exclude_cols if col in df.columns]

    # Get all numerical columns and filter out excluded ones
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

    print(f"Scaling {len(numerical_cols)} numerical features")

    # Create a copy for scaled features
    df_scaled = df.copy()

    # Initialize scaler
    scaler = StandardScaler()

    # Fit and transform
    if len(numerical_cols) > 0:
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df_scaled


def analyze_features(df, target_col='turnout_percentage'):
    """Analyze feature correlations with target."""
    print("\nAnalyzing feature correlations...")

    if target_col not in df.columns:
        print(f"Target column {target_col} not found in dataframe.")
        return

    # Calculate correlations with target
    try:
        correlations = df.select_dtypes(include=['int64', 'float64']).corr()[target_col]
        correlations = correlations.sort_values(ascending=False)

        # Print top positive correlations
        print("\nTop 10 positive correlations with turnout:")
        print(correlations.head(10))

        # Print top negative correlations
        print("\nTop 10 negative correlations with turnout:")
        print(correlations.tail(10))

        # Create correlation heatmap
        plt.figure(figsize=(12, 10))

        # Select top features by absolute correlation
        top_features = correlations.drop(target_col).abs().sort_values(ascending=False).head(10).index

        # Create correlation matrix with these features
        corr_matrix = df[list(top_features) + [target_col]].corr()

        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Top Feature Correlations with Voter Turnout')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'))
        plt.close()

        print(f"Saved correlation heatmap to 'reports/figures/correlation_heatmap.png'")

    except Exception as e:
        print(f"Error analyzing correlations: {e}")


def main():
    """Main function to process data."""
    print("Direct Data Processing for Voter Turnout Prediction")
    print("==================================================")

    try:
        # Merge datasets
        print("\n1. Merging datasets")
        merged_df = merge_datasets()

        if merged_df is None or len(merged_df) == 0:
            print("Error: Could not create merged dataset.")
            return

        print(f"\nSuccessfully created dataset with {len(merged_df)} rows and {len(merged_df.columns)} columns.")

        # Preview the data
        print("\nPreview of merged data:")
        print(merged_df.head(3))

        # Create features
        print("\n2. Creating features")
        featured_df = create_features(merged_df)

        # Handle outliers
        print("\n3. Handling outliers")
        featured_df = handle_outliers(featured_df)

        # Encode categorical features
        print("\n4. Encoding categorical features")
        encoded_df = encode_categorical_features(featured_df)

        # Scale features
        print("\n5. Scaling features")
        scaled_df = scale_features(encoded_df)

        # Analyze features
        print("\n6. Analyzing feature correlations")
        analyze_features(encoded_df)

        # Save processed data
        processed_file = os.path.join(processed_dir, 'voter_turnout_features.csv')
        encoded_df.to_csv(processed_file, index=False)
        print(f"\nSaved processed data to {processed_file}")

        scaled_file = os.path.join(processed_dir, 'voter_turnout_features_scaled.csv')
        scaled_df.to_csv(scaled_file, index=False)
        print(f"Saved scaled data to {scaled_file}")

        print("\nData processing complete!")
        print(f"Final dataset has {len(encoded_df)} rows and {len(encoded_df.columns)} columns.")
        print(f"Ready for model training.")

    except Exception as e:
        print(f"Error in main processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()