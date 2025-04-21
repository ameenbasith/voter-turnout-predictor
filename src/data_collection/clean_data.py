#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
data_dir = '../data/raw'  # or wherever you put your files
output_dir = '../data/analysis'
os.makedirs(output_dir, exist_ok=True)

# Load files
census = pd.read_csv(os.path.join(data_dir, 'census_county_data_2020.csv'))
elections = pd.read_csv(os.path.join(data_dir, 'countypres_2000-2020.csv'))

# Prepare census data
census['fips'] = census['fips'].astype(str).str.zfill(5)
census['higher_education_percentage'] = census['higher_education'] / census['total_population'] * 100

# Prepare election data
elections = elections[(elections['year'] == 2020) & (elections['office'] == 'US PRESIDENT')]
elections = elections[['county_fips', 'candidate', 'party', 'candidatevotes', 'totalvotes']]

# Drop missing FIPS
elections = elections.dropna(subset=['county_fips'])
elections['county_fips'] = elections['county_fips'].astype(int).astype(str).str.zfill(5)

# Simplify to two parties
elections['party_simple'] = elections['party'].apply(lambda x: 'DEMOCRAT' if x == 'DEMOCRAT' else ('REPUBLICAN' if x == 'REPUBLICAN' else 'OTHER'))

# Aggregate votes
election_summary = elections.pivot_table(
    index='county_fips',
    columns='party_simple',
    values='candidatevotes',
    aggfunc='sum'
).fillna(0).reset_index()

# Calculate total votes again if necessary
election_summary['totalvotes'] = election_summary['DEMOCRAT'] + election_summary['REPUBLICAN'] + election_summary['OTHER']

# Calculate margins
election_summary['margin'] = (election_summary['DEMOCRAT'] - election_summary['REPUBLICAN']) / election_summary['totalvotes'] * 100

# Merge education and election
merged = pd.merge(
    census[['fips', 'higher_education_percentage']],
    election_summary[['county_fips', 'margin']],
    left_on='fips',
    right_on='county_fips',
    how='inner'
)

# Analyze
correlation = merged['higher_education_percentage'].corr(merged['margin'])
print(f"\n✅ Correlation between Higher Education % and Democratic Margin: {correlation:.4f}")

# Save merged file
merged.to_csv(os.path.join(output_dir, 'education_vs_partisan_lean.csv'), index=False)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged, x='higher_education_percentage', y='margin')
sns.regplot(data=merged, x='higher_education_percentage', y='margin', scatter=False, color='black')
plt.axhline(0, color='gray', linestyle='--')
plt.title(f'Higher Education % vs Democratic Margin (2020)\nCorrelation = {correlation:.2f}', fontsize=16)
plt.xlabel('Higher Education Percentage')
plt.ylabel('Partisan Margin (% Dem - % Rep)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'education_vs_partisan_lean_2020.png'))
plt.show()
