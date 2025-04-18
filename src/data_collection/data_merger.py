#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed data merger script that correctly aligns county-level census data
with county-level election data for voter turnout prediction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
print("Setting up directories...")
raw_dir = '../data/raw'
processed_dir = '../data/processed'
figures_dir = '../reports/figures'

# Create directories if they don't exist
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)


def load_election_data():
    """
    Load and preprocess election data from countypres_2000-2020.csv.
    Focus on presidential elections for 2012, 2016, and 2020.
    """
    print("\nLoading election data...")

    # Path to election data
    election_file = os.path.join(raw_dir, 'countypres_2000-2020.csv')

    # Check if file exists
    if not os.path.exists(election_file):
        raise FileNotFoundError(f"Election data file not found at {election_file}")

    # Load the data
    election_df = pd.read_csv(election_file)
    print(f"Loaded election data: {election_df.shape[0]} rows, {election_df.shape[1]} columns")

    # Filter to the years we're interested in
    election_df = election_df[election_df['year'].isin([2012, 2016, 2020])]
    print(f"Filtered to 2012, 2016, 2020: {election_df.shape[0]} rows")

    # Filter to TOTAL mode (aggregate across voting methods)
    if 'mode' in election_df.columns:
        election_df = election_df[election_df['mode'] == 'TOTAL']
        print(f"Filtered to TOTAL mode: {election_df.shape[0]} rows")

    # Check for missing FIPS codes
    missing_fips = election_df['county_fips'].isna().sum()
    if missing_fips > 0:
        print(f"Found {missing_fips} rows with missing FIPS codes - removing these")
        election_df = election_df.dropna(subset=['county_fips'])

    # Convert FIPS codes to strings with 5 digits
    election_df['county_fips'] = election_df['county_fips'].astype(float).astype(int).astype(str).str.zfill(5)

    # Aggregate votes by county
    # Sum votes by candidate, then sum across candidates
    county_votes = election_df.groupby(['year', 'state', 'state_po', 'county_name', 'county_fips']).agg({
        'candidatevotes': 'sum',
        'totalvotes': 'first'  # Should be the same for all candidates in a county
    }).reset_index()

    # Rename columns
    county_votes = county_votes.rename(columns={
        'candidatevotes': 'total_votes_cast',
        'totalvotes': 'total_votes',
        'state_po': 'state_abbr'
    })

    # Extract state and county parts of FIPS code
    county_votes['state_fips'] = county_votes['county_fips'].str[:2]
    county_votes['county_fips_only'] = county_votes['county_fips'].str[2:]

    print(f"Prepared county-level election data: {county_votes.shape[0]} rows")
    return county_votes

def load_and_process_individual_census_files():
    """
    Load cleaned census data directly from the raw directory (user moved files here).
    """
    print("\nLoading cleaned census data...")

    clean_dir = "../data/raw"  # updated to match your moved location
    census_dfs = []

    for year in [2012, 2016, 2020]:
        filename = f"census_county_data_{year}_clean.csv"
        year_file = os.path.join(clean_dir, filename)
        if os.path.exists(year_file):
            df = pd.read_csv(year_file)
            if 'year' not in df.columns:
                df['year'] = year
            census_dfs.append(df)
            print(f"✅ Loaded cleaned census data for {year}: {df.shape[0]} rows")
        else:
            print(f"❌ Cleaned file not found for year {year}: {year_file}")

    if not census_dfs:
        raise FileNotFoundError("No valid cleaned census data files found")

    census_df = pd.concat(census_dfs, ignore_index=True)
    print(f"\n📊 Combined cleaned census data: {census_df.shape[0]} rows")

    # Ensure FIPS codes are properly formatted
    census_df['county_fips'] = census_df['fips'].astype(str).str.zfill(5)

    # Add state abbreviations
    state_fips_to_abbr = {
        '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT',
        '10': 'DE', '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL',
        '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD',
        '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE',
        '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
        '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD',
        '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV',
        '55': 'WI', '56': 'WY'
    }
    census_df['state_fips'] = census_df['county_fips'].str[:2]
    census_df['state_abbr'] = census_df['state_fips'].map(state_fips_to_abbr)

    # Calculate higher education percentage if missing
    if 'higher_education_percentage' not in census_df.columns and 'higher_education' in census_df.columns:
        census_df['higher_education_percentage'] = (
            census_df['higher_education'] / census_df['total_population']) * 100
        print("✅ Created higher_education_percentage")

    return census_df


def create_swing_state_data():
    """
    Create data identifying swing states for each election year.
    """
    print("\nCreating swing state data...")

    # Define swing states for each election
    swing_states = {
        2012: ['FL', 'OH', 'NC', 'VA', 'WI', 'CO', 'IA', 'NH', 'NV'],
        2016: ['FL', 'PA', 'OH', 'MI', 'NC', 'WI', 'AZ', 'CO', 'IA', 'NH', 'NV'],
        2020: ['FL', 'PA', 'MI', 'NC', 'WI', 'AZ', 'GA', 'MN', 'NH', 'NV']
    }

    # Create dataframe
    swing_data = []
    for year, states in swing_states.items():
        for state in states:
            swing_data.append({
                'year': year,
                'state_abbr': state,
                'is_swing_state': True
            })

    swing_df = pd.DataFrame(swing_data)
    print(f"Created swing state data with {len(swing_df)} entries")

    return swing_df


def simulate_ad_spending(merged_df):
    """
    Simulate campaign ad spending based on county population and swing state status.
    """
    print("\nSimulating campaign ad spending...")

    # Make a copy to avoid modifying the input
    df = merged_df.copy()

    # Set base spending per capita for each election year
    base_spending = {
        2012: 2.5,  # dollars per person
        2016: 3.0,
        2020: 3.5
    }

    # Set swing state multiplier
    swing_multiplier = 5.0  # 5x more spending in swing states

    # Set education effect multiplier
    education_multiplier = 0.05  # 5% increase per percentage point above 20%

    # Create random multiplier for variation
    np.random.seed(42)  # For reproducibility
    df['random_factor'] = np.random.lognormal(mean=0, sigma=0.4, size=len(df))

    # Calculate ad spending
    def calculate_ad_spend(row):
        # Get base rate for the year
        base_rate = base_spending.get(row['year'], 3.0)

        # Adjust for swing state
        swing_factor = swing_multiplier if row.get('is_swing_state', False) else 1.0

        # Adjust for education (targeting more educated areas)
        education_factor = 1.0
        if 'higher_education_percentage' in row and not pd.isna(row['higher_education_percentage']):
            education_bonus = max(0, row['higher_education_percentage'] - 20) * education_multiplier
            education_factor = 1.0 + education_bonus

        # Calculate population-based spending
        population = row.get('total_population', 100000)
        if pd.isna(population) or population <= 0:
            population = 100000  # Default if missing

        # Calculate total spending
        total_spend = population * base_rate * swing_factor * education_factor * row['random_factor']

        # Calculate party-specific spending
        if row.get('is_swing_state', False):
            # More equal spending in swing states
            rep_share = np.random.uniform(0.45, 0.55)
        else:
            # More variable in non-swing states
            rep_share = np.random.uniform(0.35, 0.65)

        rep_spend = total_spend * rep_share
        dem_spend = total_spend * (1 - rep_share)

        return {
            'total_ad_spend': total_spend,
            'republican_ad_spend': rep_spend,
            'democrat_ad_spend': dem_spend
        }

    # Apply the calculation
    ad_data = df.apply(calculate_ad_spend, axis=1, result_type='expand')

    # Add columns to the dataframe
    df['total_ad_spend'] = ad_data['total_ad_spend'].round(0)
    df['republican_ad_spend'] = ad_data['republican_ad_spend'].round(0)
    df['democrat_ad_spend'] = ad_data['democrat_ad_spend'].round(0)

    # Calculate ad spend per capita
    df['ad_spend_per_capita'] = (df['total_ad_spend'] / df['total_population']).round(2)

    # Remove temporary column
    df = df.drop(columns=['random_factor'])

    print(f"Added simulated ad spending data to {len(df)} counties")
    return df


def merge_datasets():
    """
    Merge election, census, and swing state data into a single dataset.
    """
    print("\nMerging datasets...")

    # Load the individual datasets
    election_df = load_election_data()
    census_df = load_and_process_individual_census_files()
    swing_df = create_swing_state_data()

    # Check alignment between election and census data
    election_counties = set(election_df['county_fips'])
    census_counties = set(census_df['county_fips'])
    common_counties = election_counties.intersection(census_counties)

    print(f"\nData alignment check:")
    print(f"Election data: {len(election_counties)} unique counties")
    print(f"Census data: {len(census_counties)} unique counties")
    print(
        f"Common counties: {len(common_counties)} ({len(common_counties) / len(election_counties) * 100:.1f}% coverage)")

    # Show some examples of common counties to verify
    common_example = list(common_counties)[:5]
    print(f"Sample common counties (FIPS): {common_example}")

    # Check a few counties in both datasets to ensure alignment
    print("\nVerifying a few counties in both datasets:")
    for fips in common_example:
        e_data = election_df[election_df['county_fips'] == fips].iloc[0]
        c_data = census_df[census_df['county_fips'] == fips].iloc[0]
        print(
            f"FIPS {fips}: {e_data['county_name']} - Election: {e_data['total_votes_cast']:,} votes, Census: {c_data['total_population']:,} population")

    # Merge election data with swing state data
    print("\nMerging election data with swing state data...")
    merged_df = pd.merge(
        election_df,
        swing_df,
        on=['year', 'state_abbr'],
        how='left'
    )

    # Fill missing swing state values with False
    merged_df['is_swing_state'] = merged_df['is_swing_state'].fillna(False)

    # Check the merge result
    swing_count = merged_df['is_swing_state'].sum()
    print(f"Merged dataset has {len(merged_df)} rows with {swing_count} from swing states")

    # Merge with census data on county_fips and year
    print("\nMerging with census data...")
    final_df = pd.merge(
        merged_df,
        census_df,
        on=['county_fips', 'year'],
        how='inner',
        suffixes=('', '_census')
    )

    # Check the merge result
    print(f"Final merged dataset has {len(final_df)} rows")

    # Calculate voter turnout
    print("\nCalculating voter turnout percentage...")
    if 'total_population' in final_df.columns and 'total_votes_cast' in final_df.columns:
        # Voting age population is approximately 75% of total population
        final_df['voting_age_population'] = final_df['total_population'] * 0.75

        # Calculate turnout as percentage of voting age population
        final_df['turnout_percentage'] = (final_df['total_votes_cast'] / final_df['voting_age_population']) * 100

        # Cap turnout at 100%
        final_df['turnout_percentage'] = final_df['turnout_percentage'].clip(upper=100)

        # Check turnout distribution
        turnout_bins = [0, 20, 40, 60, 80, 100]
        turnout_counts = pd.cut(final_df['turnout_percentage'], bins=turnout_bins).value_counts().sort_index()
        print("\nTurnout percentage distribution:")
        for i, count in enumerate(turnout_counts):
            if i < len(turnout_bins) - 1:
                print(
                    f"  {turnout_bins[i]}% to {turnout_bins[i + 1]}%: {count} counties ({count / len(final_df) * 100:.1f}%)")

        # Check for unreasonable turnout values (very high or very low)
        high_turnout = (final_df['turnout_percentage'] > 90).sum()
        low_turnout = (final_df['turnout_percentage'] < 20).sum()

        if high_turnout > 0:
            print(f"Note: {high_turnout} counties ({high_turnout / len(final_df) * 100:.1f}%) have turnout > 90%")
        if low_turnout > 0:
            print(f"Note: {low_turnout} counties ({low_turnout / len(final_df) * 100:.1f}%) have turnout < 20%")
    else:
        print("Warning: Cannot calculate turnout - missing required columns")

    # Add simulated ad spending
    final_df = simulate_ad_spending(final_df)

    # Clean up columns
    # Remove duplicate columns from the merge
    duplicate_cols = [col for col in final_df.columns if col.endswith('_census')]
    final_df = final_df.drop(columns=duplicate_cols)

    # Drop any rows with missing turnout percentage
    missing_turnout = final_df['turnout_percentage'].isna().sum()
    if missing_turnout > 0:
        print(f"Dropping {missing_turnout} rows with missing turnout percentage")
        final_df = final_df.dropna(subset=['turnout_percentage'])

    # Calculate average turnout by year
    yearly_turnout = final_df.groupby('year')['turnout_percentage'].mean()
    print("\nAverage turnout by election year:")
    for year, turnout in yearly_turnout.items():
        print(f"  {year}: {turnout:.2f}%")

    # Calculate average turnout by swing state status
    swing_turnout = final_df.groupby(['year', 'is_swing_state'])['turnout_percentage'].mean().unstack()
    print("\nAverage turnout by swing state status:")
    for year in swing_turnout.index:
        print(
            f"  {year}: Swing States: {swing_turnout.loc[year, True]:.2f}%, Non-Swing States: {swing_turnout.loc[year, False]:.2f}%")

    # Final dataset information
    print(f"\nFinal dataset has {len(final_df)} rows and {len(final_df.columns)} columns")
    print(f"Elections covered: {final_df['year'].unique()}")
    print(f"States covered: {final_df['state_abbr'].nunique()}")
    print(f"Counties covered: {final_df['county_fips'].nunique()}")

    # Return the merged dataset
    return final_df


def visualize_merged_data(df):
    """
    Create exploratory visualizations of the merged dataset.
    """
    print("\nCreating exploratory visualizations...")

    # Create directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Turnout distribution by year
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='year', y='turnout_percentage', data=df)
    plt.title('Voter Turnout by Election Year')
    plt.xlabel('Election Year')
    plt.ylabel('Turnout Percentage')
    plt.savefig(os.path.join(figures_dir, 'turnout_by_year.png'))
    plt.close()

    # 2. Turnout by swing state status
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='year', y='turnout_percentage', hue='is_swing_state', data=df)
    plt.title('Voter Turnout: Swing vs. Non-Swing States')
    plt.xlabel('Election Year')
    plt.ylabel('Turnout Percentage')
    plt.legend(title='Swing State')
    plt.savefig(os.path.join(figures_dir, 'turnout_by_swing_state.png'))
    plt.close()

    # 3. Education vs. Turnout
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='higher_education_percentage', y='turnout_percentage',
                    hue='year', size='total_population', sizes=(20, 200),
                    alpha=0.6, data=df)
    plt.title('Higher Education vs. Voter Turnout')
    plt.xlabel('Higher Education Percentage')
    plt.ylabel('Turnout Percentage')
    # Add trendline
    sns.regplot(x='higher_education_percentage', y='turnout_percentage',
                data=df, scatter=False, color='black')
    plt.savefig(os.path.join(figures_dir, 'education_vs_turnout.png'))
    plt.close()

    # 4. Income vs. Turnout
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='median_household_income', y='turnout_percentage',
                    hue='year', size='total_population', sizes=(20, 200),
                    alpha=0.6, data=df)
    plt.title('Median Income vs. Voter Turnout')
    plt.xlabel('Median Household Income ($)')
    plt.ylabel('Turnout Percentage')
    # Add trendline
    sns.regplot(x='median_household_income', y='turnout_percentage',
                data=df, scatter=False, color='black')
    plt.savefig(os.path.join(figures_dir, 'income_vs_turnout.png'))
    plt.close()

    # 5. Ad Spending vs. Turnout
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='ad_spend_per_capita', y='turnout_percentage',
                    hue='is_swing_state', size='total_population', sizes=(20, 200),
                    alpha=0.6, data=df)
    plt.title('Ad Spending Per Capita vs. Voter Turnout')
    plt.xlabel('Ad Spending Per Capita ($)')
    plt.ylabel('Turnout Percentage')
    # Add trendline
    sns.regplot(x='ad_spend_per_capita', y='turnout_percentage',
                data=df, scatter=False, color='black')
    plt.savefig(os.path.join(figures_dir, 'ad_spending_vs_turnout.png'))
    plt.close()

    # 6. Correlations with turnout
    plt.figure(figsize=(12, 10))
    numeric_cols = ['total_population', 'median_household_income',
                    'higher_education_percentage', 'unemployment_rate',
                    'ad_spend_per_capita', 'turnout_percentage']
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Key Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'correlation_matrix.png'))
    plt.close()

    # 7. Create a new visualization: Population vs Turnout
    plt.figure(figsize=(10, 6))
    # Use log scale for population as it's often skewed
    plt.scatter(df['total_population'], df['turnout_percentage'],
                alpha=0.5, c=df['higher_education_percentage'], cmap='viridis')
    plt.xscale('log')
    plt.colorbar(label='Higher Education Percentage')
    plt.title('County Population vs. Voter Turnout')
    plt.xlabel('County Population (log scale)')
    plt.ylabel('Turnout Percentage')
    plt.savefig(os.path.join(figures_dir, 'population_vs_turnout.png'))
    plt.close()

    print(f"Saved 7 exploratory visualizations to {figures_dir}")


def main():
    """
    Main function to merge datasets and create a dataset for modeling.
    """
    print("Voter Turnout Predictor - Fixed Data Preparation")
    print("===============================================")

    try:
        # Merge datasets
        final_df = merge_datasets()

        # Visualize the merged data
        visualize_merged_data(final_df)

        # Save the merged dataset
        output_file = os.path.join(processed_dir, 'voter_turnout_dataset_fixed.csv')
        final_df.to_csv(output_file, index=False)
        print(f"\nSaved merged dataset to {output_file}")

        print("\nData preparation complete!")
        print("You can now proceed with feature engineering and model training.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()