#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Further analysis: National turnout trends
"""

import pandas as pd
import os

# Paths to your raw data
election_path = 'data/raw/countypres_2000-2020.csv'
census_2012_path = 'data/raw/census_county_data_2012.csv'
census_2016_path = 'data/raw/census_county_data_2016.csv'
census_2020_path = 'data/raw/census_county_data_2020.csv'

# Load election data
elections = pd.read_csv(election_path)
elections = elections[elections['office'] == 'US PRESIDENT']
elections = elections[elections['year'].isin([2012, 2016, 2020])]
elections = elections[elections['mode'] == 'TOTAL']

# Clean FIPS codes
elections = elections.dropna(subset=['county_fips'])
elections['county_fips'] = elections['county_fips'].astype(int).astype(str).str.zfill(5)

# Sum total votes per county
county_votes = elections.groupby(['year', 'county_fips']).agg({
    'totalvotes': 'first'
}).reset_index()

# Load census data
census_2012 = pd.read_csv(census_2012_path)
census_2016 = pd.read_csv(census_2016_path)
census_2020 = pd.read_csv(census_2020_path)

# Clean FIPS
for census_df in [census_2012, census_2016, census_2020]:
    census_df['fips'] = census_df['fips'].astype(str).str.zfill(5)

# Add year column
census_2012['year'] = 2012
census_2016['year'] = 2016
census_2020['year'] = 2020

# Combine census data
census = pd.concat([census_2012, census_2016, census_2020], ignore_index=True)

# Merge on year + FIPS
merged = pd.merge(county_votes, census, left_on=['year', 'county_fips'], right_on=['year', 'fips'])

# ✅ FIX: Estimate Voting Age Population (VAP)
merged['voting_age_population'] = merged['total_population'] * 0.75

# ✅ Calculate turnout based on VAP
merged['turnout_percentage'] = (merged['totalvotes'] / merged['voting_age_population']) * 100

# Calculate national turnout by year
national_turnout = merged.groupby('year')['turnout_percentage'].mean().reset_index()
national_turnout.rename(columns={'turnout_percentage': 'national_turnout_pct'}, inplace=True)

print("\n=== National Turnout by Year ===")
print(national_turnout)

# Calculate change from 2016 to 2020
turnout_2016 = national_turnout.loc[national_turnout['year'] == 2016, 'national_turnout_pct'].values[0]
turnout_2020 = national_turnout.loc[national_turnout['year'] == 2020, 'national_turnout_pct'].values[0]
turnout_change = (turnout_2020 - turnout_2016) / turnout_2016 * 100

print(f"\n✅ National Turnout Change from 2016 to 2020: {turnout_change:.2f}%")

# Save results
os.makedirs('data/analysis', exist_ok=True)
national_turnout.to_csv('data/analysis/national_turnout_by_year.csv', index=False)

print("\n✅ Saved national turnout analysis!")
