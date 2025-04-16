#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script simulates campaign ad spending data by county based on population
and swing state status, since actual detailed county-level ad spending data
may be difficult to obtain freely.
"""

import os
import pandas as pd
import numpy as np
import random


def load_county_data():
    """
    Load county demographic data to use as a basis for simulation.

    Returns:
        pandas.DataFrame: County demographic data
    """
    # Check if the file exists
    file_path = 'data/raw/census_county_data_all_years.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Run census_data_collector.py first.")

    return pd.read_csv(file_path)


def identify_swing_states():
    """
    Identify swing states for weighting ad spending appropriately.

    Returns:
        list: List of swing state abbreviations
    """
    # Historical swing states in recent elections
    swing_states_2012 = ['FL', 'OH', 'NC', 'VA', 'WI', 'CO', 'IA', 'NH', 'NV']
    swing_states_2016 = ['FL', 'PA', 'OH', 'MI', 'NC', 'WI', 'AZ', 'CO', 'IA', 'NH', 'NV']
    swing_states_2020 = ['FL', 'PA', 'MI', 'NC', 'WI', 'AZ', 'GA', 'MN', 'NH', 'NV']

    # Dictionary of swing states by year
    swing_states = {
        2012: swing_states_2012,
        2016: swing_states_2016,
        2020: swing_states_2020
    }

    return swing_states


def simulate_ad_spending(county_df):
    """
    Simulate campaign ad spending based on population, swing state status,
    and some randomness to mimic real-world variations.

    Args:
        county_df (pandas.DataFrame): County demographic data

    Returns:
        pandas.DataFrame: Simulated ad spending data by county
    """
    print("Simulating campaign ad spending data...")

    # Get swing states by year
    swing_states = identify_swing_states()

    # Create a copy of the input data to avoid modifying it
    df = county_df.copy()

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

    df['state_abbr'] = df['state_fips'].map(state_fips_to_abbr)

    # Function to determine if a state is a swing state in a given year
    def is_swing_state(row):
        year = row['year']
        state_abbr = row['state_abbr']
        return state_abbr in swing_states.get(year, [])

    # Add swing state indicator
    df['is_swing_state'] = df.apply(is_swing_state, axis=1)

    # Base spending per capita (in dollars)
    base_spending_per_capita = {
        2012: 2.5,
        2016: 3.0,
        2020: 3.5
    }

    # Swing state multiplier
    swing_state_multiplier = 5  # Spend 5x more in swing states

    # Add some randomness to make it more realistic
    df['ad_spend_multiplier'] = np.random.lognormal(mean=0, sigma=0.5, size=len(df))

    # Calculate ad spending
    def calculate_ad_spend(row):
        year = row['year']
        base_rate = base_spending_per_capita.get(year, 3.0)
        multiplier = swing_state_multiplier if row['is_swing_state'] else 1
        random_factor = row['ad_spend_multiplier']

        # Calculate based on population
        population = row['total_population']
        if pd.isna(population) or population <= 0:
            return 0

        # More spending in counties with higher education (targeted ads)
        education_factor = 1.0
        if not pd.isna(row['higher_education']) and row['higher_education'] > 0:
            education_percentage = row['higher_education'] / population
            education_factor = 1.0 + education_percentage

        # Base calculation
        ad_spend = population * base_rate * multiplier * random_factor * education_factor

        # Add some political party bias (Republicans vs Democrats spending patterns)
        if random.random() > 0.5:  # Randomly assign higher Republican or Democrat spending
            rep_spend = ad_spend * random.uniform(0.8, 1.2)
            dem_spend = ad_spend * random.uniform(0.8, 1.2)
        else:
            rep_spend = ad_spend * random.uniform(0.8, 1.2)
            dem_spend = ad_spend * random.uniform(0.8, 1.2)

        return {
            'total_ad_spend': ad_spend,
            'republican_ad_spend': rep_spend,
            'democrat_ad_spend': dem_spend
        }

    # Apply the calculation
    ad_spend_data = df.apply(calculate_ad_spend, axis=1)

    # Extract the results
    df['total_ad_spend'] = ad_spend_data.apply(lambda x: x['total_ad_spend'])
    df['republican_ad_spend'] = ad_spend_data.apply(lambda x: x['republican_ad_spend'])
    df['democrat_ad_spend'] = ad_spend_data.apply(lambda x: x['democrat_ad_spend'])

    # Round to nearest dollar
    df['total_ad_spend'] = df['total_ad_spend'].round(0)
    df['republican_ad_spend'] = df['republican_ad_spend'].round(0)
    df['democrat_ad_spend'] = df['democrat_ad_spend'].round(0)

    # Select relevant columns for output
    output_columns = [
        'year', 'state', 'state_abbr', 'county', 'county_name', 'fips',
        'is_swing_state', 'total_ad_spend', 'republican_ad_spend', 'democrat_ad_spend'
    ]

    return df[output_columns]


def main():
    """Main function to simulate and save campaign ad spending data."""
    # Create data directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    try:
        # Load county demographic data
        county_df = load_county_data()

        # Simulate ad spending
        ad_spending_df = simulate_ad_spending(county_df)

        # Save the simulated data
        output_file = 'data/processed/simulated_campaign_ad_spending.csv'
        ad_spending_df.to_csv(output_file, index=False)
        print(f"Saved simulated ad spending data to {output_file}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()