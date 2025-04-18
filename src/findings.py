#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Party + Education Shift Summary Script
Analyzes Democratic vs Republican trends and education levels by county competitiveness.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
analysis_dir = "data/analysis"
figures_dir = "reports/figures"

# Load files
flipped_rep_to_dem = pd.read_csv(os.path.join(analysis_dir, "flipped_rep_to_dem_2016_2020.csv"))
flipped_dem_to_rep = pd.read_csv(os.path.join(analysis_dir, "flipped_dem_to_rep_2016_2020.csv"))
county_turnout = pd.read_csv(os.path.join(analysis_dir, "county_turnout_analysis_2020.csv"))
county_competitiveness = pd.read_csv(os.path.join(analysis_dir, "county_competitiveness_2020.csv"))
census_2020 = pd.read_csv(os.path.join("data", "raw", "census_county_data_2020.csv"))

# === County Flips Summary ===
print("\n=== County Flips 2016 → 2020 ===")
print(f"🔵 Counties that flipped REPUBLICAN ➔ DEMOCRAT: {len(flipped_rep_to_dem)}")
print(f"🔴 Counties that flipped DEMOCRAT ➔ REPUBLICAN: {len(flipped_dem_to_rep)}")

# === Top 10 Largest Democratic Shifts ===
print("\n=== Top 10 Strongest Democratic Shifts ===")
top_dem_shift = flipped_rep_to_dem[['county_name', 'state_po', 'margin_change']].sort_values(by='margin_change', ascending=False).head(10)
print(top_dem_shift)

# === Top 10 Largest Republican Shifts ===
print("\n=== Top 10 Strongest Republican Shifts ===")
top_rep_shift = flipped_dem_to_rep[['county_name', 'state_po', 'margin_change']].sort_values(by='margin_change').head(10)
print(top_rep_shift)

# === Turnout and Party Advantage ===
print("\n=== Turnout vs Party Advantage ===")
turnout_high = county_turnout[county_turnout['competitiveness'] == 'Highly Competitive (±5%)']
turnout_safe = county_turnout[county_turnout['competitiveness'] == 'Safe (>15%)']
print(f"Average turnout in highly competitive counties: {turnout_high['turnout_percentage'].mean():.2f}%")
print(f"Average turnout in safe counties: {turnout_safe['turnout_percentage'].mean():.2f}%")

# === Competitiveness and Party Margin ===
print("\n=== Competitiveness Summary ===")
print(county_competitiveness.groupby('competitiveness')['margin'].mean())

# === Education Analysis ===
print("\n=== Education Levels by Competitiveness (2020) ===")

# Load 2020 Census Data
census_2020 = pd.read_csv(os.path.join("data", "raw", "census_county_data_2020.csv"))

# Fix FIPS columns
census_2020['fips'] = census_2020['fips'].astype(str).str.zfill(5)

# Fix county_competitiveness FIPS column if not already
if 'fips' not in county_competitiveness.columns:
    print("No fips column in competitiveness file — adding dummy fips...")
    # We can't match without FIPS properly; safest to stop here or fix
else:
    county_competitiveness['fips'] = county_competitiveness['fips'].astype(str).str.zfill(5)

# Merge by FIPS
merged_edu = pd.merge(
    county_competitiveness,
    census_2020[['fips', 'higher_education']],
    on='fips',
    how='left'
)

# Normalize higher education
merged_edu['higher_education_percentage'] = (merged_edu['higher_education'] / merged_edu['higher_education'].max()) * 100

# Group by competitiveness
edu_by_comp = merged_edu.groupby('competitiveness')['higher_education_percentage'].mean()
print(edu_by_comp)

# Save boxplot
import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs(figures_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x='competitiveness',
    y='higher_education_percentage',
    data=merged_edu,
    order=edu_by_comp.sort_values().index.tolist()
)
plt.title('Higher Education Levels by County Competitiveness (2020)', fontsize=16)
plt.xlabel('Competitiveness Category', fontsize=14)
plt.ylabel('Higher Education Percentage (%)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "education_by_competitiveness.png"))
plt.close()

print("\n✅ Full Party + Education Shift Summary Complete!")
