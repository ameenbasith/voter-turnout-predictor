#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reporting module for political trends analysis.
Contains functions for generating summary reports and output files.
"""

import os
import logging
import pandas as pd
import numpy as np


def generate_summary_report(recent_changes, merged_data, counties_2020, output_dir):
    """
    Generate a summary report of key findings.
    """
    logging.info("Generating summary report...")
    output_file = os.path.join(output_dir, 'summary_report.txt')

    with open(output_file, 'w') as f:
        f.write("Political Trends and Demographic Insights Summary Report\n")
        f.write("=====================================================\n\n")

        # National vote totals
        try:
            dem_votes_2016 = merged_data[(merged_data['year'] == 2016)]['DEMOCRAT'].sum()
            rep_votes_2016 = merged_data[(merged_data['year'] == 2016)]['REPUBLICAN'].sum()
            total_votes_2016 = merged_data[(merged_data['year'] == 2016)]['totalvotes'].sum()

            dem_votes_2020 = merged_data[(merged_data['year'] == 2020)]['DEMOCRAT'].sum()
            rep_votes_2020 = merged_data[(merged_data['year'] == 2020)]['REPUBLICAN'].sum()
            total_votes_2020 = merged_data[(merged_data['year'] == 2020)]['totalvotes'].sum()

            f.write(
                f"2016 Results: D {dem_votes_2016 / total_votes_2016 * 100:.1f}%, R {rep_votes_2016 / total_votes_2016 * 100:.1f}%\n")
            f.write(
                f"2020 Results: D {dem_votes_2020 / total_votes_2020 * 100:.1f}%, R {rep_votes_2020 / total_votes_2020 * 100:.1f}%\n")
            f.write(
                f"Vote Change: D {(dem_votes_2020 / total_votes_2020 * 100) - (dem_votes_2016 / total_votes_2016 * 100):+.1f}%, ")
            f.write(
                f"R {(rep_votes_2020 / total_votes_2020 * 100) - (rep_votes_2016 / total_votes_2016 * 100):+.1f}%\n")
            f.write(f"Turnout Change: {((total_votes_2020 / total_votes_2016) - 1) * 100:+.1f}%\n\n")
        except:
            f.write("Could not calculate national vote totals due to missing data\n\n")

        # County shifts
        try:
            dem_shift = recent_changes[recent_changes['margin_change'] > 0].shape[0]
            rep_shift = recent_changes[recent_changes['margin_change'] < 0].shape[0]

            f.write(f"Counties with Democratic shift: {dem_shift} ({dem_shift / len(recent_changes) * 100:.1f}%)\n")
            f.write(f"Counties with Republican shift: {rep_shift} ({rep_shift / len(recent_changes) * 100:.1f}%)\n")

            # County flips
            flipped_dem_to_rep = recent_changes[(recent_changes['margin_from'] > 0) & (recent_changes['margin_to'] < 0)]
            flipped_rep_to_dem = recent_changes[(recent_changes['margin_from'] < 0) & (recent_changes['margin_to'] > 0)]

            f.write(f"Counties flipped Dem→Rep: {len(flipped_dem_to_rep)}, Rep→Dem: {len(flipped_rep_to_dem)}\n\n")
        except:
            f.write("Could not calculate county shifts due to missing data\n\n")

        # Key correlations
        f.write("KEY CORRELATIONS\n")
        f.write("--------------\n")

        if counties_2020 is not None:
            try:
                if 'higher_education_percentage' in counties_2020.columns and 'margin' in counties_2020.columns:
                    edu_margin_corr = counties_2020['higher_education_percentage'].corr(counties_2020['margin'])
                    f.write(f"Education and partisan lean (2020): {edu_margin_corr:.3f}\n")

                if 'median_household_income' in counties_2020.columns and 'margin' in counties_2020.columns:
                    income_margin_corr = counties_2020['median_household_income'].corr(counties_2020['margin'])
                    f.write(f"Income and partisan lean (2020): {income_margin_corr:.3f}\n")

                if 'population_density' in counties_2020.columns and 'margin' in counties_2020.columns:
                    density_margin_corr = counties_2020['population_density'].corr(counties_2020['margin'])
                    f.write(f"Population density and partisan lean (2020): {density_margin_corr:.3f}\n")
            except:
                f.write("Could not calculate demographic correlations due to missing data\n")

        try:
            turnout_margin_corr = recent_changes['turnout_change_pct'].corr(recent_changes['margin_change'])
            f.write(f"Turnout change and partisan shift (2016-2020): {turnout_margin_corr:.3f}\n\n")
        except:
            f.write("Could not calculate turnout-shift correlation due to missing data\n\n")

        f.write("ANALYSIS COMPLETED\n")
        f.write("See data/analysis/ for detailed CSVs and reports/figures/ for visualizations.\n")

    logging.info(f"Summary report saved to {output_file}")


def generate_battleground_report(merged_data, election_changes, output_dir):
    """
    Generate a report focused on battleground/swing counties.
    """
    logging.info("Generating battleground county report...")

    # Define competitive counties (margin within ±5 points)
    counties_2020 = merged_data[merged_data['year'] == 2020].copy()
    counties_2020['competitiveness'] = pd.cut(
        counties_2020['margin'].abs(),
        bins=[0, 5, 15, float('inf')],
        labels=['Highly Competitive (±5%)', 'Competitive (±5-15%)', 'Safe (>15%)']
    )
    competitive_counties = counties_2020[counties_2020['competitiveness'] == 'Highly Competitive (±5%)']

    # Find counties that flipped between 2016 and 2020
    recent_changes = election_changes[election_changes['from_year'] == 2016].copy()
    flipped_dem_to_rep = recent_changes[(recent_changes['margin_from'] > 0) & (recent_changes['margin_to'] < 0)]
    flipped_rep_to_dem = recent_changes[(recent_changes['margin_from'] < 0) & (recent_changes['margin_to'] > 0)]

    all_flipped = pd.concat([flipped_dem_to_rep, flipped_rep_to_dem])

    # Find competitive counties that also flipped
    battleground_counties = pd.merge(
        competitive_counties[['fips', 'county_name', 'state_po', 'margin']],
        all_flipped[['county_fips', 'margin_from', 'margin_to', 'margin_change']],
        left_on='fips',
        right_on='county_fips',
        how='inner'
    )

    output_file = os.path.join(output_dir, 'battleground_counties_report.txt')

    with open(output_file, 'w') as f:
        f.write("Battleground Counties Analysis\n")
        f.write("============================\n\n")

        f.write(f"Highly competitive counties (±5% margin): {len(competitive_counties)}\n")
        f.write(f"Counties that flipped parties: {len(all_flipped)}\n")
        f.write(f"Battleground counties (competitive + flipped): {len(battleground_counties)}\n\n")

        # Top battleground counties
        if len(battleground_counties) > 0:
            battleground_counties['abs_margin'] = battleground_counties['margin'].abs()
            top_battlegrounds = battleground_counties.sort_values('abs_margin').head(10)

            f.write("TOP BATTLEGROUND COUNTIES\n")
            f.write("------------------------\n")
            for idx, row in top_battlegrounds.iterrows():
                direction = "D+" if row['margin'] > 0 else "R+"
                prev = "D+" if row['margin_from'] > 0 else "R+"
                f.write(f"{row['county_name']}, {row['state_po']}: {prev}{abs(row['margin_from']):.1f} → ")
                f.write(f"{direction}{abs(row['margin']):.1f} ({row['margin_change']:+.1f})\n")

        # Save to CSV
        battleground_counties.to_csv(os.path.join(output_dir, 'battleground_counties.csv'), index=False)

    logging.info(f"Battleground counties report saved to {output_file}")
    return battleground_counties


def generate_state_report(merged_data, election_changes, state_code, output_dir):
    """
    Generate a report for a specific state's counties.
    """
    logging.info(f"Generating report for {state_code}...")

    # Filter to specified state
    state_data = merged_data[merged_data['state_po'] == state_code].copy()

    if len(state_data) == 0:
        logging.warning(f"No data found for state {state_code}")
        return

    # Get most recent election data
    most_recent_year = state_data['year'].max()
    counties_recent = state_data[state_data['year'] == most_recent_year].copy()

    # Get election changes
    state_changes = election_changes[election_changes['state_po'] == state_code].copy()
    recent_changes = state_changes[state_changes['from_year'] == 2016].copy()

    output_file = os.path.join(output_dir, f'state_report_{state_code}.txt')

    with open(output_file, 'w') as f:
        f.write(f"Political Analysis for {state_code}\n")
        f.write("=" * 30 + "\n\n")

        # State results
        try:
            dem_votes = counties_recent['DEMOCRAT'].sum()
            rep_votes = counties_recent['REPUBLICAN'].sum()
            total_votes = counties_recent['totalvotes'].sum()

            dem_share = dem_votes / total_votes * 100
            rep_share = rep_votes / total_votes * 100
            margin = dem_share - rep_share

            f.write(f"State results ({most_recent_year}): ")
            if margin > 0:
                f.write(f"D+{margin:.1f} (D: {dem_share:.1f}%, R: {rep_share:.1f}%)\n\n")
            else:
                f.write(f"R+{abs(margin):.1f} (D: {dem_share:.1f}%, R: {rep_share:.1f}%)\n\n")
        except:
            f.write("Could not calculate state results\n\n")

        # County summary
        f.write(f"Counties: {len(counties_recent)}\n")

        # Competitiveness
        counties_recent['competitiveness'] = pd.cut(
            counties_recent['margin'].abs(),
            bins=[0, 5, 15, float('inf')],
            labels=['Highly Competitive (±5%)', 'Competitive (±5-15%)', 'Safe (>15%)']
        )

        competitive_summary = counties_recent.groupby('competitiveness').size()
        for category, count in competitive_summary.items():
            f.write(f"{category}: {count} counties\n")

        # Partisan shifts
        if len(recent_changes) > 0:
            dem_shift = recent_changes[recent_changes['margin_change'] > 0].shape[0]
            rep_shift = recent_changes[recent_changes['margin_change'] < 0].shape[0]

            f.write(f"\nDemocratic shift: {dem_shift} counties, ")
            f.write(f"Republican shift: {rep_shift} counties\n")

            # Counties that flipped
            flipped_dem_to_rep = recent_changes[(recent_changes['margin_from'] > 0) & (recent_changes['margin_to'] < 0)]
            flipped_rep_to_dem = recent_changes[(recent_changes['margin_from'] < 0) & (recent_changes['margin_to'] > 0)]

            f.write(f"Flipped Dem→Rep: {len(flipped_dem_to_rep)}, ")
            f.write(f"Rep→Dem: {len(flipped_rep_to_dem)}\n")

    logging.info(f"State report saved to {output_file}")

    # Save state counties data to CSV
    counties_recent.to_csv(os.path.join(output_dir, f'state_counties_{state_code}_{most_recent_year}.csv'), index=False)