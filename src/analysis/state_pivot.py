# educational_recalculation.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
education_path = '../data/processed/state_education_summary.csv'
election_path = '../data/processed/state_pivot_summary.csv'

print("✅ Loading datasets...")

# Load datasets
education = pd.read_csv(education_path)
election = pd.read_csv(election_path)

# Map numeric state FIPS to state abbreviations
state_fips_to_abbr = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE', 11: 'DC', 12: 'FL',
    13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA',
    23: 'ME', 24: 'MD', 25: 'MA', 26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE',
    32: 'NV', 33: 'NH', 34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK',
    41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN', 48: 'TX', 49: 'UT', 50: 'VT',
    51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY', 60: 'AS', 66: 'GU', 69: 'MP', 72: 'PR',
    74: 'UM', 78: 'VI'
}
education['state_po'] = education['state'].map(state_fips_to_abbr)

# Create new column for total higher education percentage
education['higher_education_total_pct'] = (
        education['bachelors_degree_pct'] +
        education['masters_degree_pct'] +
        education['professional_degree_pct'] +
        education['doctorate_degree_pct']
)

print(f"🎓 Education data shape: {education.shape}")
print(f"🗳️ Election data shape: {election.shape}")

# Merge datasets on year and state_po
merged = pd.merge(
    education,
    election,
    how='left',
    on=['year', 'state_po']
)

print(f"✅ Merged dataset: {merged.shape}")

# Create output directory if it doesn't exist
os.makedirs('../reports/figures', exist_ok=True)

# Analyze year by year
for year in [2012, 2016, 2020]:
    df_year = merged[merged['year'] == year].dropna(subset=['higher_education_total_pct', 'dem_margin'])

    if df_year.empty:
        print(f"⚠️ No data available for {year}. Skipping...")
        continue

    correlation = df_year['higher_education_total_pct'].corr(df_year['dem_margin'])

    print(f"\n📅 Year {year}:")
    print(f"📈 Correlation between Total Higher Education % and Democratic Margin: {correlation:.4f}")

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df_year['higher_education_total_pct'],
        df_year['dem_margin'],
        alpha=0.7
    )
    plt.title(f'Higher Education % vs Democratic Margin ({year})')
    plt.xlabel('Higher Education %')
    plt.ylabel('Democratic Margin (%)')
    plt.grid(True)
    plt.tight_layout()

    plot_path = f'reports/figures/education_vs_margin_{year}.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"🖼️ Saved scatter plot to {plot_path}")

print("\n✅ Full analysis complete!")
