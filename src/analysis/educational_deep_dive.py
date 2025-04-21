import pandas as pd

# === Load data ===
edu = pd.read_csv('../data/raw/census_county_data_all_years.csv')
election = pd.read_csv('../data/raw/countypres_2000-2020.csv')
edu_state = pd.read_csv('../data/processed/state_education_summary.csv')  # 🔥 Correct location now!

print("✅ Loaded education:", edu.shape)
print("✅ Loaded election:", election.shape)
print("✅ Education state summary loaded:", edu_state.shape)

# === Prepare election data ===
election['county_fips'] = election['county_fips'].astype(str).str.zfill(5)

# Create 'party_simplified' for election data
election['party_simplified'] = 'OTHER'
election.loc[election['party'].str.contains('DEMOCRAT', case=False, na=False), 'party_simplified'] = 'DEMOCRAT'
election.loc[election['party'].str.contains('REPUBLICAN', case=False, na=False), 'party_simplified'] = 'REPUBLICAN'
election.loc[(election['party_simplified'] == 'OTHER') &
             (election['candidate'].str.contains('BIDEN|OBAMA|CLINTON', case=False, na=False)), 'party_simplified'] = 'DEMOCRAT'
election.loc[(election['party_simplified'] == 'OTHER') &
             (election['candidate'].str.contains('TRUMP|ROMNEY', case=False, na=False)), 'party_simplified'] = 'REPUBLICAN'

# Focus only on general elections for president
election = election[(election['office'] == 'US PRESIDENT') & (election['mode'] == 'TOTAL')]

# === Aggregate to state-year-party level ===
state_votes = election.groupby(['year', 'state_po', 'party_simplified'])['candidatevotes'].sum().reset_index()

# Pivot so each row is state-year, columns are DEMOCRAT, REPUBLICAN, OTHER
state_pivot = state_votes.pivot(index=['year', 'state_po'], columns='party_simplified', values='candidatevotes').fillna(0).reset_index()

# Calculate margin
state_pivot['dem_margin'] = (state_pivot['DEMOCRAT'] - state_pivot['REPUBLICAN']) / (state_pivot['DEMOCRAT'] + state_pivot['REPUBLICAN']) * 100

print("\n✅ State pivot ready:")
print(state_pivot.head())

# === Map state codes to abbreviations ===
state_map = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE', 11: 'DC', 12: 'FL', 13: 'GA',
    15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD',
    25: 'MA', 26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH', 34: 'NJ',
    35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC',
    46: 'SD', 47: 'TN', 48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY'
}

edu_state['state_po'] = edu_state['state'].map(state_map)

# === Merge education with election results ===
merged = pd.merge(edu_state, state_pivot, how='left', left_on=['year', 'state_po'], right_on=['year', 'state_po'])

print("\n✅ Merged data:")
print(merged.head())

# === Correlation analysis ===
for year in [2012, 2016, 2020]:
    subset = merged[merged['year'] == year]
    if not subset.empty:
        corr = subset['bachelors_degree_pct'].corr(subset['dem_margin'])
        print(f"📈 {year} - Correlation between Bachelors Degree % and Democratic Margin: {corr:.4f}")
    else:
        print(f"⚠️ {year} - No data for {year}")
