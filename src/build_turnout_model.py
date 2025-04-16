import pandas as pd
import numpy as np
import os

# Load data
census_path = "data/raw/census_county_data_all_years.csv"
election_path = "data/raw/countypres_2000-2020.csv"

# Load datasets
census_df = pd.read_csv(census_path)
election_df = pd.read_csv(election_path)

# Clean & prepare FIPS codes
census_df['county_fips'] = census_df['fips'].astype(str).str.zfill(5)
election_df['county_fips'] = election_df['county_fips'].dropna().astype(int).astype(str).str.zfill(5)

# Ensure year is the same type
census_df['year'] = census_df['year'].astype(int)
election_df['year'] = election_df['year'].astype(int)

# Filter election data for relevant years and mode TOTAL if it exists
election_df = election_df[election_df['year'].isin([2012, 2016, 2020])]
if 'mode' in election_df.columns:
    election_df = election_df[election_df['mode'] == 'TOTAL']

# Aggregate election data by county and year
vote_agg = election_df.groupby(['year', 'county_fips'], as_index=False).agg({
    'candidatevotes': 'sum',
    'totalvotes': 'first',
    'state': 'first',
    'county_name': 'first'
})
vote_agg = vote_agg.rename(columns={
    'candidatevotes': 'total_votes_cast',
    'totalvotes': 'total_votes_reported',
    'state': 'state_abbr'
})

# Merge datasets on year and county_fips
merged_df = pd.merge(vote_agg, census_df, on=['year', 'county_fips'], how='left')

# Estimate Voting Age Population (VAP) and calculate turnout
merged_df['estimated_vap'] = merged_df['total_population'] * 0.75
merged_df['turnout_percentage'] = (merged_df['total_votes_cast'] / merged_df['estimated_vap']) * 100
merged_df['turnout_percentage'] = merged_df['turnout_percentage'].clip(upper=100)

# Create higher education percentage if education columns exist
edu_cols = ['bachelors_degree', 'masters_degree', 'professional_degree', 'doctorate_degree']
if all(col in merged_df.columns for col in edu_cols):
    merged_df['higher_education'] = merged_df[edu_cols].sum(axis=1)
    merged_df['higher_education_percentage'] = (merged_df['higher_education'] / merged_df['total_population']) * 100
    merged_df['higher_education_percentage'] = merged_df['higher_education_percentage'].clip(upper=100)

# Simulate swing state status
swing_states = {
    2012: ['FL', 'OH', 'NC', 'VA', 'WI', 'CO', 'IA', 'NH', 'NV'],
    2016: ['FL', 'PA', 'OH', 'MI', 'NC', 'WI', 'AZ', 'CO', 'IA', 'NH', 'NV'],
    2020: ['FL', 'PA', 'MI', 'NC', 'WI', 'AZ', 'GA', 'MN', 'NH', 'NV']
}
merged_df['is_swing_state'] = merged_df.apply(
    lambda row: row['state_abbr'] in swing_states.get(row['year'], []),
    axis=1
)

# Simulate ad spending based on population, year, swing state, and random variation
np.random.seed(42)
base_spending = {2012: 2.5, 2016: 3.0, 2020: 3.5}
swing_multiplier = 5

merged_df['ad_spend_multiplier'] = np.random.lognormal(mean=0, sigma=0.5, size=len(merged_df))
merged_df['base_rate'] = merged_df['year'].map(base_spending)

merged_df['total_ad_spend'] = (
    merged_df['total_population'] *
    merged_df['base_rate'] *
    merged_df['ad_spend_multiplier'] *
    merged_df['is_swing_state'].apply(lambda x: swing_multiplier if x else 1)
)

merged_df['republican_ad_spend'] = merged_df['total_ad_spend'] * np.random.uniform(0.45, 0.55, size=len(merged_df))
merged_df['democrat_ad_spend'] = merged_df['total_ad_spend'] * np.random.uniform(0.45, 0.55, size=len(merged_df))

# Save cleaned + merged dataset for modeling
os.makedirs("data/processed", exist_ok=True)
merged_df.to_csv("data/processed/merged_turnout_data.csv", index=False)

print("✅ Merged dataset saved! Rows:", len(merged_df))