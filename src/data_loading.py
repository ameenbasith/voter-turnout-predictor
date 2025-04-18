#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading and preprocessing module for political trends analysis.
"""

import os
import logging
import pandas as pd
import numpy as np


def load_census_data(data_dir):
    """
    Load and preprocess census data from 2012, 2016, and 2020.
    Focus on relative changes rather than absolute values.

    Args:
        data_dir (str): Directory containing raw data files

    Returns:
        pandas.DataFrame: Processed census data
    """
    logging.info("Loading census data...")

    census_files = {
        2012: os.path.join(data_dir, 'census_county_data_2012.csv'),
        2016: os.path.join(data_dir, 'census_county_data_2016.csv'),
        2020: os.path.join(data_dir, 'census_county_data_2020.csv')
    }

    census_dfs = {}
    for year, file_path in census_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Ensure FIPS codes are strings with leading zeros
            df['fips'] = df['fips'].astype(str).str.zfill(5)
            census_dfs[year] = df
            logging.info(f"Loaded {year} census data: {len(df)} counties")

    if not census_dfs:
        raise FileNotFoundError("No census data files found")

    # Combine into a single dataframe
    census_df = pd.concat(census_dfs.values(), ignore_index=True)
    logging.info(f"Combined census data: {len(census_df)} counties across {len(census_dfs)} years")

    # Calculate normalized values for comparison
    # Group by county to enable analysis of changes over time
    counties = census_df.groupby('fips')

    # Create a list to store processed county data
    processed_counties = []

    for fips, county_data in counties:
        if len(county_data) > 1:  # Only process counties with data for multiple years
            # Sort by year
            county_data = county_data.sort_values('year')

            # Create a copy for this county
            county_processed = county_data.copy()

            # Calculate education metrics
            # - Percent change in higher education from baseline (2012 or earliest available)
            if 'higher_education' in county_data.columns:
                baseline_edu = county_data['higher_education'].iloc[0]
                if baseline_edu > 0:  # Avoid division by zero
                    county_processed['higher_education_pct_change'] = (county_data[
                                                                           'higher_education'] / baseline_edu - 1) * 100

            # Unemployment rate change from baseline
            if 'unemployment_rate' in county_data.columns:
                baseline_unemp = county_data['unemployment_rate'].iloc[0]
                county_processed['unemployment_rate_change'] = county_data['unemployment_rate'] - baseline_unemp

            # Income growth from baseline
            if 'median_household_income' in county_data.columns:
                baseline_income = county_data['median_household_income'].iloc[0]
                if baseline_income > 0:  # Avoid division by zero
                    county_processed['income_pct_change'] = (county_data[
                                                                 'median_household_income'] / baseline_income - 1) * 100

            # Add to our processed data
            processed_counties.append(county_processed)

    # Combine processed county data
    if processed_counties:
        census_processed = pd.concat(processed_counties, ignore_index=True)
        logging.info(f"Processed census data with change metrics: {len(census_processed)} counties")
        return census_processed
    else:
        logging.warning("No counties with multi-year data found. Using original census data.")
        return census_df


def load_election_data(data_dir):
    """
    Load and preprocess election data.
    Calculate Democratic and Republican vote shares and changes over time.

    Args:
        data_dir (str): Directory containing raw data files

    Returns:
        tuple: (county_results, election_changes) DataFrames
    """
    logging.info("Loading election data...")

    # Try different possible file names for election data
    possible_files = [
        os.path.join(data_dir, 'countypres_2000-2020.csv'),
        os.path.join(data_dir, 'countypres_20002020.csv'),
        os.path.join(data_dir, 'mit_election_lab_county_returns_raw.csv')
    ]

    election_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            election_file = file_path
            break

    if not election_file:
        raise FileNotFoundError(
            "No election data file found in the data/raw directory. Please make sure one of these files exists: " +
            ", ".join(os.path.basename(f) for f in possible_files))

    # Load the election data
    election_df = pd.read_csv(election_file)
    logging.info(f"Loaded election data: {election_df.shape[0]} rows, {election_df.shape[1]} columns")

    # Filter to presidential elections for 2012, 2016, and 2020
    election_df = election_df[(election_df['year'].isin([2012, 2016, 2020])) &
                              (election_df['office'] == 'US PRESIDENT')]

    # Filter to general election results (TOTAL mode)
    if 'mode' in election_df.columns:
        election_df = election_df[election_df['mode'] == 'TOTAL']

    logging.info(f"Filtered to presidential elections 2012-2020: {election_df.shape[0]} rows")

    # Check for missing FIPS codes
    missing_fips = election_df['county_fips'].isna().sum()
    if missing_fips > 0:
        logging.warning(f"Found {missing_fips} rows with missing FIPS codes - removing these")
        election_df = election_df.dropna(subset=['county_fips'])

    # Convert FIPS codes to strings with 5 digits
    election_df['county_fips'] = election_df['county_fips'].astype(float).astype(int).astype(str).str.zfill(5)

    # Process to get Democratic and Republican vote shares
    # First, identify Democratic and Republican candidates
    dem_candidates = ['BARACK OBAMA', 'HILLARY CLINTON', 'JOSEPH R BIDEN JR', 'BIDEN, JOSEPH R JR']
    rep_candidates = ['MITT ROMNEY', 'DONALD TRUMP', 'DONALD J TRUMP']

    # Create party indicator
    election_df['party_simplified'] = 'OTHER'
    for candidate in dem_candidates:
        election_df.loc[
            election_df['candidate'].str.contains(candidate, case=False, na=False), 'party_simplified'] = 'DEMOCRAT'
    for candidate in rep_candidates:
        election_df.loc[
            election_df['candidate'].str.contains(candidate, case=False, na=False), 'party_simplified'] = 'REPUBLICAN'

    # If we have a party column, use it as backup
    if 'party' in election_df.columns:
        election_df.loc[(election_df['party_simplified'] == 'OTHER') &
                        (election_df['party'] == 'DEMOCRAT'), 'party_simplified'] = 'DEMOCRAT'
        election_df.loc[(election_df['party_simplified'] == 'OTHER') &
                        (election_df['party'] == 'REPUBLICAN'), 'party_simplified'] = 'REPUBLICAN'

    # Aggregate votes by county, year, and simplified party
    county_results = election_df.groupby(['year', 'county_fips', 'county_name', 'state_po', 'party_simplified']) \
        .agg({'candidatevotes': 'sum', 'totalvotes': 'first'}) \
        .reset_index()

    # Pivot to get Dem and GOP votes in separate columns
    county_wide = county_results.pivot_table(
        index=['year', 'county_fips', 'county_name', 'state_po', 'totalvotes'],
        columns='party_simplified',
        values='candidatevotes',
        aggfunc='sum'
    ).reset_index()

    # Fill NaN values with 0 (counties where a party had no votes)
    if 'DEMOCRAT' not in county_wide.columns:
        county_wide['DEMOCRAT'] = 0
    if 'REPUBLICAN' not in county_wide.columns:
        county_wide['REPUBLICAN'] = 0
    if 'OTHER' not in county_wide.columns:
        county_wide['OTHER'] = 0

    # Calculate vote shares
    county_wide['dem_share'] = county_wide['DEMOCRAT'] / county_wide['totalvotes'] * 100
    county_wide['rep_share'] = county_wide['REPUBLICAN'] / county_wide['totalvotes'] * 100
    county_wide['other_share'] = county_wide['OTHER'] / county_wide['totalvotes'] * 100

    # Calculate two-party vote share
    two_party_total = county_wide['DEMOCRAT'] + county_wide['REPUBLICAN']
    county_wide['dem_share_twoparty'] = county_wide['DEMOCRAT'] / two_party_total * 100
    county_wide['rep_share_twoparty'] = county_wide['REPUBLICAN'] / two_party_total * 100

    # Calculate margin (positive = Dem advantage, negative = Rep advantage)
    county_wide['margin'] = county_wide['dem_share'] - county_wide['rep_share']
    county_wide['margin_twoparty'] = county_wide['dem_share_twoparty'] - county_wide['rep_share_twoparty']

    logging.info(f"Processed election data: {len(county_wide)} county-elections")

    # Now calculate changes over time for each county
    county_changes = []

    for county_fips in county_wide['county_fips'].unique():
        county_data = county_wide[county_wide['county_fips'] == county_fips].sort_values('year')

        if len(county_data) > 1:  # Need at least two elections to calculate change
            for i in range(1, len(county_data)):
                prev_year = county_data.iloc[i - 1]['year']
                curr_year = county_data.iloc[i]['year']

                # Create a row for this county's changes
                change_row = {
                    'county_fips': county_fips,
                    'county_name': county_data.iloc[i]['county_name'],
                    'state_po': county_data.iloc[i]['state_po'],
                    'from_year': prev_year,
                    'to_year': curr_year,
                    'dem_share_from': county_data.iloc[i - 1]['dem_share'],
                    'dem_share_to': county_data.iloc[i]['dem_share'],
                    'rep_share_from': county_data.iloc[i - 1]['rep_share'],
                    'rep_share_to': county_data.iloc[i]['rep_share'],
                    'margin_from': county_data.iloc[i - 1]['margin'],
                    'margin_to': county_data.iloc[i]['margin'],
                    'totalvotes_from': county_data.iloc[i - 1]['totalvotes'],
                    'totalvotes_to': county_data.iloc[i]['totalvotes'],
                    'dem_share_change': county_data.iloc[i]['dem_share'] - county_data.iloc[i - 1]['dem_share'],
                    'rep_share_change': county_data.iloc[i]['rep_share'] - county_data.iloc[i - 1]['rep_share'],
                    'margin_change': county_data.iloc[i]['margin'] - county_data.iloc[i - 1]['margin'],
                    'turnout_change_pct': (county_data.iloc[i]['totalvotes'] / county_data.iloc[i - 1][
                        'totalvotes'] - 1) * 100
                }

                county_changes.append(change_row)

    # Convert to DataFrame
    changes_df = pd.DataFrame(county_changes)
    logging.info(f"Calculated changes for {len(changes_df)} county-election pairs")

    return county_wide, changes_df


def merge_data(election_df, census_df):
    """
    Merge election and census data to analyze relationships.

    Args:
        election_df (pandas.DataFrame): Election data
        census_df (pandas.DataFrame): Census data

    Returns:
        pandas.DataFrame: Merged dataset
    """
    logging.info("Merging election and census data...")

    # Prepare for merge
    election_df = election_df.copy()
    census_df = census_df.copy()

    # Make sure FIPS codes are consistent
    election_df['fips'] = election_df['county_fips']

    # Merge based on FIPS code and year
    merged_df = pd.merge(
        election_df,
        census_df,
        on=['fips', 'year'],
        how='inner',
        suffixes=('_election', '_census')
    )

    logging.info(f"Merged dataset has {len(merged_df)} rows")
    return merged_df