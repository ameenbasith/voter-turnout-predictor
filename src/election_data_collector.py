#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script downloads and processes county-level presidential election data
from the MIT Election Data and Science Lab via Harvard Dataverse.
"""

import os
import pandas as pd
import requests
import zipfile
import io
import time


def download_election_data():
    """
    Load the county-level presidential election returns from local file.

    Returns:
        pandas.DataFrame: Election returns by county
    """
    # Check if the file already exists locally
    local_file_path = 'data/raw/countypres_2000-2020.csv'

    if os.path.exists(local_file_path):
        print(f"Loading election data from local file: {local_file_path}")
        return pd.read_csv(local_file_path)

    # If the file doesn't exist locally, we'll try to download it
    print("Local file not found. Attempting to download from Harvard Dataverse...")

    # Direct download URL for the dataset
    url = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/VOQCHQ/IQNIFU"

    # Make the request and download the file
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download data: Status code {response.status_code}")

        # The file is a TSV file
        print("Download successful! Processing the data...")
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep='\t')

        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please ensure the file 'countypres_2000-2020.csv' is in the data/raw directory.")
        raise


def process_election_data(df):
    """
    Process the raw election data to extract turnout information.

    Args:
        df (pandas.DataFrame): Raw election data

    Returns:
        pandas.DataFrame: Processed election data with turnout metrics
    """
    print("Processing election data...")

    # Filter to include only general elections (not primaries)
    if 'mode' in df.columns:
        df = df[df['mode'] == 'TOTAL']

    # Filter to include only the years we're interested in
    target_years = [2012, 2016, 2020]
    df = df[df['year'].isin(target_years)]

    # Calculate total votes per county per election
    groupby_cols = ['year', 'state', 'state_po', 'county_name', 'county_fips']
    # Ensure all columns exist
    groupby_cols = [col for col in groupby_cols if col in df.columns]

    # Different versions of the dataset may have different column names
    if 'candidatevotes' in df.columns and 'totalvotes' in df.columns:
        turnout_df = df.groupby(groupby_cols).agg({
            'candidatevotes': 'sum',
            'totalvotes': 'first'  # Total votes should be the same for all candidates in a county
        }).reset_index()

        # Rename columns for clarity
        turnout_df = turnout_df.rename(columns={
            'candidatevotes': 'total_votes_cast',
            'totalvotes': 'total_votes_reported'
        })
    elif 'votes' in df.columns:
        # If dataset has different structure, adapt accordingly
        turnout_df = df.groupby(groupby_cols).agg({
            'votes': 'sum'
        }).reset_index()

        # Rename columns for clarity
        turnout_df = turnout_df.rename(columns={
            'votes': 'total_votes_cast'
        })
        turnout_df['total_votes_reported'] = turnout_df['total_votes_cast']
    else:
        raise Exception("Dataset format not recognized - please check column names")

    # Rename state_po to state_abbr for consistency
    if 'state_po' in turnout_df.columns:
        turnout_df = turnout_df.rename(columns={'state_po': 'state_abbr'})

    # Some rows might have missing FIPS codes
    if 'county_fips' in turnout_df.columns:
        turnout_df = turnout_df.dropna(subset=['county_fips'])

        # Convert FIPS to string with leading zeros
        turnout_df['county_fips'] = turnout_df['county_fips'].astype(str).str.zfill(5)

        # Add state FIPS and county FIPS separately
        turnout_df['state_fips'] = turnout_df['county_fips'].str[:2]
        turnout_df['county_fips_only'] = turnout_df['county_fips'].str[2:]

    return turnout_df


def main():
    """Main function to collect and process election data."""
    # Create data directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    try:
        # Download the data
        election_df = download_election_data()

        # Save the raw data
        election_df.to_csv('data/raw/mit_election_lab_county_returns_raw.csv', index=False)
        print("Saved raw election data to data/raw/mit_election_lab_county_returns_raw.csv")

        # Process the data to get turnout information
        turnout_df = process_election_data(election_df)

        # Save the processed turnout data
        turnout_df.to_csv('data/processed/county_election_turnout.csv', index=False)
        print("Saved processed turnout data to data/processed/county_election_turnout.csv")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()