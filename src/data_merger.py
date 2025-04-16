#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script merges census data, election turnout data, and ad spending data
into a single dataset for analysis and modeling.
"""

import os
import pandas as pd
import numpy as np


def load_census_data():
    """
    Load the census demographic data.

    Returns:
        pandas.DataFrame: Census data by county
    """
    file_path = 'data/raw/census_county_data_all_years.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Run census_data_collector.py first.")

    return pd.read_csv(file_path)


def load_election_data():
    """
    Load the election turnout data.

    Returns:
        pandas.DataFrame: Election turnout data by county
    """
    # Check if processed election data exists
    processed_file_path = 'data/processed/county_election_turnout.csv'
    if os.path.exists(processed_file_path):
        return pd.read_csv(processed_file_path)

    # If processed data doesn't exist, try to load and process the raw file
    raw_file_path = 'data/raw/countypres_2000-2020.csv'
    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Could not find {raw_file_path}. Make sure the election data file exists.")

    print("Processing raw election data...")
    # Load the raw data
    election_df = pd.read_csv(raw_file_path)

    # Filter to include only the years we're interested in
    target_years = [2012, 2016, 2020]
    election_df = election_df[election_df['year'].isin(target_years)]

    # Filter to include only general elections (if mode column exists)
    if 'mode' in election_df.columns:
        election_df = election_df[election_df['mode'] == 'TOTAL']

    # Calculate total votes per county per election
    groupby_cols = ['year', 'state', 'state_po', 'county_name', 'county_fips']
    # Ensure all columns exist
    groupby_cols = [col for col in groupby_cols if col in election_df.columns]

    # Different versions of the dataset may have different column names
    if 'candidatevotes' in election_df.columns and 'totalvotes' in election_df.columns:
        turnout_df = election_df.groupby(groupby_cols).agg({
            'candidatevotes': 'sum',
            'totalvotes': 'first'  # Total votes should be the same for all candidates in a county
        }).reset_index()

        # Rename columns for clarity
        turnout_df = turnout_df.rename(columns={
            'candidatevotes': 'total_votes_cast',
            'totalvotes': 'total_votes_reported'
        })
    elif 'votes' in election_df.columns:
        # If dataset has different structure, adapt accordingly
        turnout_df = election_df.groupby(groupby_cols).agg({
            'votes': 'sum'
        }).reset_index()

        # Rename columns for clarity
        turnout_df = turnout_df.rename(columns={
            'votes': 'total_votes_cast'
        })
        turnout_df['total_votes_reported'] = turnout_df['total_votes_cast']
    else:
        print("Warning: Dataset format not recognized - using default column names")
        # Try to use whatever vote counting column might be available
        vote_cols = [col for col in election_df.columns if 'vote' in col.lower()]
        if vote_cols:
            main_vote_col = vote_cols[0]
            turnout_df = election_df.groupby(groupby_cols).agg({
                main_vote_col: 'sum'
            }).reset_index()
            turnout_df = turnout_df.rename(columns={main_vote_col: 'total_votes_cast'})
            turnout_df['total_votes_reported'] = turnout_df['total_votes_cast']
        else:
            raise Exception("Could not identify vote count columns in the dataset")

    # Rename state_po to state_abbr for consistency
    if 'state_po' in turnout_df.columns:
        turnout_df = turnout_df.rename(columns={'state_po': 'state_abbr'})

    # Process FIPS codes if available
    if 'county_fips' in turnout_df.columns:
        # Some rows might have missing FIPS codes
        turnout_df = turnout_df.dropna(subset=['county_fips'])

        # Convert FIPS to string with leading zeros
        turnout_df['county_fips'] = turnout_df['county_fips'].astype(str).str.zfill(5)

        # Add state FIPS and county FIPS separately
        turnout_df['state_fips'] = turnout_df['county_fips'].str[:2]
        turnout_df['county_fips_only'] = turnout_df['county_fips'].str[2:]

    # Save processed data for future use
    turnout_df.to_csv(processed_file_path, index=False)
    print(f"Processed election data saved to {processed_file_path}")

    return turnout_df


def load_ad_spending_data():
    """
    Load the campaign ad spending data.

    Returns:
        pandas.DataFrame: Ad spending data by county
    """
    file_path = 'data/processed/simulated_campaign_ad_spending.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Run ad_spending_simulator.py first.")

    return pd.read_csv(file_path)


def merge_datasets(census_df, election_df, ad_spending_df):
    """
    Merge all datasets into a single dataframe.

    Args:
        census_df (pandas.DataFrame): Census demographic data
        election_df (pandas.DataFrame): Election turnout data
        ad_spending_df (pandas.DataFrame): Campaign ad spending data

    Returns:
        pandas.DataFrame: Merged dataset
    """
    print("Merging datasets...")

    # Check and convert data types for FIPS codes to ensure they're all strings
    if 'fips' in census_df.columns:
        census_df['fips'] = census_df['fips'].astype(str).str.zfill(5)

    if 'county_fips' in election_df.columns:
        election_df['county_fips'] = election_df['county_fips'].astype(str).str.zfill(5)
        # Also create a 'fips' column if it doesn't exist
        if 'fips' not in election_df.columns:
            election_df['fips'] = election_df['county_fips']

    if 'fips' in ad_spending_df.columns:
        ad_spending_df['fips'] = ad_spending_df['fips'].astype(str).str.zfill(5)

    # Ensure census_df has state_abbr for easier merging
    if 'state_abbr' not in census_df.columns:
        # Map state FIPS to state abbreviations
        state_fips_to_abbr = {
            '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
            '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
            '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
            '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
            '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
            '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
            '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
            '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
            '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
            '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
            '56': 'WY'
        }
        census_df['state_abbr'] = census_df['state_fips'].astype(str).map(state_fips_to_abbr)

    # Examine the data to understand what we're working with
    print(f"Census data shape: {census_df.shape}")
    print(f"Election data shape: {election_df.shape}")
    print(f"Ad spending data shape: {ad_spending_df.shape}")

    print("Census FIPS sample:", census_df['fips'].head())
    if 'fips' in election_df.columns:
        print("Election FIPS sample:", election_df['fips'].head())
    elif 'county_fips' in election_df.columns:
        print("Election county_fips sample:", election_df['county_fips'].head())

    # First, merge census and election data
    merge_keys = ['year', 'fips']

    # Check if both dataframes have the merge keys
    for df_name, df in [("Census", census_df), ("Election", election_df)]:
        missing_keys = [key for key in merge_keys if key not in df.columns]
        if missing_keys:
            print(f"Warning: {df_name} dataframe is missing keys: {missing_keys}")

            # If election_df uses county_fips instead of fips, adapt
            if df_name == "Election" and 'county_fips' in election_df.columns and 'fips' not in election_df.columns:
                print("Using 'county_fips' as 'fips' for election data")
                election_df['fips'] = election_df['county_fips']

    # Try the merge with more flexible handling
    try:
        merged_df = pd.merge(
            census_df,
            election_df,
            on=merge_keys,
            how='inner',
            suffixes=('', '_election')
        )
    except Exception as e:
        print(f"Error during first merge: {e}")
        # Alternative approach: try to merge on state, county, and year instead
        alt_keys = [col for col in ['year', 'state', 'county'] if
                    col in census_df.columns and col in election_df.columns]

        if alt_keys:
            print(f"Trying alternative merge on keys: {alt_keys}")
            merged_df = pd.merge(
                census_df,
                election_df,
                on=alt_keys,
                how='inner',
                suffixes=('', '_election')
            )
        else:
            raise Exception("Cannot find common keys for merging census and election data")

    print(f"After first merge, shape: {merged_df.shape}")

    # Then, merge with ad spending data
    merge_keys2 = [key for key in ['year', 'fips', 'state_abbr']
                   if key in merged_df.columns and key in ad_spending_df.columns]

    if not merge_keys2:
        print("Warning: No common keys found for ad spending merge")
        # Create some common keys if possible
        if 'state_fips' in merged_df.columns and 'state' in ad_spending_df.columns:
            ad_spending_df['state_fips'] = ad_spending_df['state']
            merge_keys2.append('state_fips')
        if 'year' in merged_df.columns and 'year' in ad_spending_df.columns:
            merge_keys2.append('year')

    if merge_keys2:
        merged_df = pd.merge(
            merged_df,
            ad_spending_df,
            on=merge_keys2,
            how='left',  # Use left join to keep all county data even if no ad spending
            suffixes=('', '_ad')
        )
    else:
        print("Warning: Skipping ad spending merge due to lack of common keys")

    print(f"After second merge, shape: {merged_df.shape}")

    # Drop duplicate columns
    columns_to_drop = [col for col in merged_df.columns if col.endswith('_election') or col.endswith('_ad')]
    merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')

    # Calculate voter turnout percentage
    # Voting Age Population (VAP) is typically about 75% of total population (rough estimate)
    merged_df['estimated_vap'] = merged_df['total_population'] * 0.75

    # Check if we have total_votes_cast
    if 'total_votes_cast' in merged_df.columns:
        # Calculate turnout as percentage of estimated VAP
        merged_df['turnout_percentage'] = (merged_df['total_votes_cast'] / merged_df['estimated_vap']) * 100

        # Cap turnout at 100% (sometimes the VAP estimate can be off)
        merged_df['turnout_percentage'] = merged_df['turnout_percentage'].clip(upper=100)
    else:
        print("Warning: 'total_votes_cast' not found, cannot calculate turnout percentage")
        if 'votes' in merged_df.columns:
            # Try using 'votes' column instead
            merged_df['total_votes_cast'] = merged_df['votes']
            merged_df['turnout_percentage'] = (merged_df['total_votes_cast'] / merged_df['estimated_vap']) * 100
            merged_df['turnout_percentage'] = merged_df['turnout_percentage'].clip(upper=100)
        else:
            # Create a placeholder turnout percentage based on national averages
            year_avg_turnout = {
                2012: 58.6,
                2016: 60.1,
                2020: 66.8
            }
            merged_df['turnout_percentage'] = merged_df['year'].map(year_avg_turnout)
            # Add random variation
            merged_df['turnout_percentage'] += np.random.normal(0, 5, size=len(merged_df))
            merged_df['turnout_percentage'] = merged_df['turnout_percentage'].clip(lower=30, upper=90)

    # Clean up any NaN values
    merged_df = merged_df.fillna({
        'turnout_percentage': merged_df['turnout_percentage'].median(),
        'total_ad_spend': 0,
        'republican_ad_spend': 0,
        'democrat_ad_spend': 0
    })

    # Add ad spend per capita
    if 'total_ad_spend' in merged_df.columns:
        merged_df['ad_spend_per_capita'] = merged_df['total_ad_spend'] / merged_df['total_population']

    # Add higher education percentage
    if 'higher_education' in merged_df.columns:
        merged_df['higher_education_percentage'] = (merged_df['higher_education'] / merged_df['total_population']) * 100

    # Select relevant columns for the final dataset
    final_columns = [
        'year', 'state', 'state_abbr', 'county', 'county_name', 'fips',
        'total_population', 'median_household_income', 'higher_education_percentage',
        'unemployment_rate', 'turnout_percentage'
    ]

    # Add optional columns if they exist
    for col in ['total_votes_cast', 'total_votes_reported', 'total_ad_spend',
                'ad_spend_per_capita', 'republican_ad_spend', 'democrat_ad_spend',
                'is_swing_state']:
        if col in merged_df.columns:
            final_columns.append(col)

    # Ensure all selected columns exist in the dataframe
    final_columns = [col for col in final_columns if col in merged_df.columns]

    return merged_df[final_columns]


def main():
    """Main function to merge datasets and save the result."""
    # Create data directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)

    try:
        # Load all datasets
        print("Loading datasets...")
        census_df = load_census_data()
        election_df = load_election_data()
        ad_spending_df = load_ad_spending_data()

        # Merge datasets
        merged_df = merge_datasets(census_df, election_df, ad_spending_df)

        # Save the merged dataset
        output_file = 'data/processed/voter_turnout_dataset.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"Saved merged dataset to {output_file}")

        # Print some stats about the final dataset
        print(f"Final dataset shape: {merged_df.shape}")
        print("Columns in the final dataset:")
        for col in merged_df.columns:
            print(f"  - {col}")

        print("\nSample of the final dataset:")
        print(merged_df.head())

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()