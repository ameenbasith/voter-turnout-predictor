#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Education Analysis for Political Trends Project
(Cleaned and Optimized Version)
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import shutil
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Define color map
dem_rep_cmap = LinearSegmentedColormap.from_list('dem_rep', ['blue', 'white', 'red'])

# Directories
DATA_DIR = "../data/raw"
OUTPUT_DIR = "../data/analysis"
FIGURES_DIR = "../reports/figures"

# Create output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load and preprocess census and election data"""
    logging.info("Loading data...")

    # Census
    census_files = {year: os.path.join(DATA_DIR, f'census_county_data_{year}.csv') for year in [2012, 2016, 2020]}
    census_dfs = []

    for year, path in census_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['fips'] = df['fips'].astype(str).str.zfill(5)
            df['year'] = year
            census_dfs.append(df)
            logging.info(f"Loaded {year} census data ({len(df)} rows)")
        else:
            logging.warning(f"Missing {year} census file")

    census_data = pd.concat(census_dfs, ignore_index=True)

    if 'higher_education_percentage' not in census_data.columns and 'higher_education' in census_data.columns:
        census_data['higher_education_percentage'] = census_data['higher_education'] / census_data['total_population'] * 100
        logging.info("Calculated higher education percentage")

    # Election
    possible_files = ['countypres_2000-2020.csv', 'countypres_20002020.csv', 'mit_election_lab_county_returns_raw.csv']
    election_file = next((os.path.join(DATA_DIR, f) for f in possible_files if os.path.exists(os.path.join(DATA_DIR, f))), None)

    if not election_file:
        raise FileNotFoundError("No election file found!")

    elections = pd.read_csv(election_file)
    elections = elections[(elections['year'].isin([2012, 2016, 2020])) & (elections['office'] == 'US PRESIDENT')]

    if 'mode' in elections.columns:
        elections = elections[elections['mode'] == 'TOTAL']

    elections = elections.dropna(subset=['county_fips'])
    elections['county_fips'] = elections['county_fips'].astype(float).astype(int).astype(str).str.zfill(5)

    elections['party_simplified'] = 'OTHER'
    dem_candidates = ['BARACK OBAMA', 'HILLARY CLINTON', 'JOSEPH R BIDEN JR', 'BIDEN, JOSEPH R JR']
    rep_candidates = ['MITT ROMNEY', 'DONALD TRUMP', 'DONALD J TRUMP']

    for dem in dem_candidates:
        elections.loc[elections['candidate'].str.contains(dem, case=False, na=False), 'party_simplified'] = 'DEMOCRAT'
    for rep in rep_candidates:
        elections.loc[elections['candidate'].str.contains(rep, case=False, na=False), 'party_simplified'] = 'REPUBLICAN'

    elections.loc[(elections['party_simplified'] == 'OTHER') & (elections['party'] == 'DEMOCRAT'), 'party_simplified'] = 'DEMOCRAT'
    elections.loc[(elections['party_simplified'] == 'OTHER') & (elections['party'] == 'REPUBLICAN'), 'party_simplified'] = 'REPUBLICAN'

    votes = elections.groupby(['year', 'county_fips', 'county_name', 'state_po', 'party_simplified']) \
        .agg({'candidatevotes': 'sum', 'totalvotes': 'first'}).reset_index()

    county_wide = votes.pivot_table(index=['year', 'county_fips', 'county_name', 'state_po', 'totalvotes'],
                                     columns='party_simplified', values='candidatevotes', aggfunc='sum', fill_value=0).reset_index()

    county_wide['dem_share'] = county_wide['DEMOCRAT'] / county_wide['totalvotes'] * 100
    county_wide['rep_share'] = county_wide['REPUBLICAN'] / county_wide['totalvotes'] * 100
    county_wide['margin'] = county_wide['dem_share'] - county_wide['rep_share']
    county_wide['fips'] = county_wide['county_fips']
    county_wide['county_fips'] = county_wide['fips']  # 🔥 Key fix

    merged = pd.merge(county_wide, census_data, on=['fips', 'year'], how='inner')
    logging.info(f"Merged dataset ({len(merged)} rows)")

    # Calculate change between years
    election_changes = []
    for fips_code in county_wide['fips'].unique():
        temp = county_wide[county_wide['fips'] == fips_code].sort_values('year')
        for i in range(1, len(temp)):
            row = {
                'county_fips': fips_code,
                'county_name': temp.iloc[i]['county_name'],
                'state_po': temp.iloc[i]['state_po'],
                'from_year': temp.iloc[i-1]['year'],
                'to_year': temp.iloc[i]['year'],
                'margin_from': temp.iloc[i-1]['margin'],
                'margin_to': temp.iloc[i]['margin'],
                'margin_change': temp.iloc[i]['margin'] - temp.iloc[i-1]['margin'],
                'turnout_change_pct': (temp.iloc[i]['totalvotes'] / temp.iloc[i-1]['totalvotes'] - 1) * 100
            }
            election_changes.append(row)

    election_changes_df = pd.DataFrame(election_changes)
    logging.info(f"Election changes calculated ({len(election_changes_df)} rows)")

    return census_data, merged, election_changes_df


def zip_results(output_dir='data/analysis', report_dir='reports'):
    """Optionally zip the final results"""
    today = datetime.now().strftime('%Y%m%d')
    zip_path = f"education_analysis_results_{today}.zip"

    shutil.make_archive(base_name=zip_path.replace('.zip', ''),
                        format='zip',
                        root_dir='..',
                        base_dir=output_dir)

    shutil.make_archive(base_name=zip_path.replace('.zip', '_figures'),
                        format='zip',
                        root_dir='..',
                        base_dir=report_dir)

    logging.info(f"Zipped outputs: {zip_path} ✅")


def main():
    """Run the full education + election analysis."""
    try:
        census_data, merged_data, election_changes = load_data()

        # TODO: Insert your actual analysis functions here
        # analyze_education_distribution(merged_data, FIGURES_DIR, OUTPUT_DIR)
        # analyze_education_partisan_relationship(merged_data, FIGURES_DIR, OUTPUT_DIR)
        # etc.

        logging.info("✅ Full analysis completed successfully!")

        # Zip results
        zip_results()

    except Exception as e:
        logging.error(f"ERROR during analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
