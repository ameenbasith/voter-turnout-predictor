#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis module for political trends analysis.
Contains functions for analyzing various aspects of electoral data.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from config import setup_plotting

# Get plotting configuration
plot_config = setup_plotting()
dem_rep_cmap = plot_config['dem_rep_cmap']


def analyze_partisan_shifts(election_changes, figures_dir, output_dir):
    """
    Analyze which counties had the biggest partisan shifts.

    Args:
        election_changes (pandas.DataFrame): DataFrame with election changes
        figures_dir (str): Directory to save figures
        output_dir (str): Directory to save output data

    Returns:
        pandas.DataFrame: Recent changes (2016-2020)
    """
    logging.info("Analyzing partisan shifts...")

    # Focus on 2016-2020 changes
    recent_changes = election_changes[election_changes['from_year'] == 2016]

    # Sort by margin change (most Democratic shift to most Republican shift)
    dem_shift = recent_changes.sort_values('margin_change', ascending=False).head(20)
    rep_shift = recent_changes.sort_values('margin_change', ascending=True).head(20)

    logging.info("Top 20 counties with strongest Democratic shifts (2016-2020):")
    for idx, row in dem_shift.iterrows():
        logging.info(f"{row['county_name']}, {row['state_po']}: {row['margin_change']:.2f}")

    logging.info("Top 20 counties with strongest Republican shifts (2016-2020):")
    for idx, row in rep_shift.iterrows():
        logging.info(f"{row['county_name']}, {row['state_po']}: {row['margin_change']:.2f}")

    # Visualize distribution of margin changes
    plt.figure(figsize=(12, 6))
    sns.histplot(recent_changes['margin_change'], bins=50, kde=True)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('Distribution of County Partisan Shifts (2016-2020)', fontsize=16)
    plt.xlabel('Change in Margin (+ = Democratic shift, - = Republican shift)', fontsize=14)
    plt.ylabel('Number of Counties', fontsize=14)
    plt.savefig(os.path.join(figures_dir, 'partisan_shift_distribution.png'))
    plt.close()

    # Create scatter plot of 2016 margin vs 2020 margin
    counties_2016 = election_changes[election_changes['from_year'] == 2016]

    plt.figure(figsize=(10, 10))
    plt.scatter(counties_2016['margin_from'], counties_2016['margin_to'],
                alpha=0.5, c=counties_2016['margin_change'], cmap=dem_rep_cmap,
                s=counties_2016['totalvotes_to'] / 1000)
    plt.colorbar(label='Margin Change (+ = Dem shift, - = Rep shift)')

    # Add reference line (no change)
    xlim = plt.xlim()
    ylim = plt.ylim()
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    plt.title('County Partisan Margins: 2016 vs 2020', fontsize=16)
    plt.xlabel('2016 Margin (+ = Dem advantage, - = Rep advantage)', fontsize=14)
    plt.ylabel('2020 Margin (+ = Dem advantage, - = Rep advantage)', fontsize=14)

    plt.savefig(os.path.join(figures_dir, 'margin_2016_vs_2020.png'))
    plt.close()

    # Analyze counties that flipped
    flipped_dem_to_rep = recent_changes[(recent_changes['margin_from'] > 0) & (recent_changes['margin_to'] < 0)]
    flipped_rep_to_dem = recent_changes[(recent_changes['margin_from'] < 0) & (recent_changes['margin_to'] > 0)]

    logging.info(f"Counties that flipped Democratic to Republican (2016-2020): {len(flipped_dem_to_rep)}")
    logging.info(f"Counties that flipped Republican to Democratic (2016-2020): {len(flipped_rep_to_dem)}")

    # Save flipped counties to CSV
    flipped_dem_to_rep.to_csv(os.path.join(output_dir, 'flipped_dem_to_rep_2016_2020.csv'), index=False)
    flipped_rep_to_dem.to_csv(os.path.join(output_dir, 'flipped_rep_to_dem_2016_2020.csv'), index=False)

    return recent_changes


def analyze_turnout_changes(election_changes, figures_dir):
    """
    Analyze turnout changes and their relationship with partisan shifts.

    Args:
        election_changes (pandas.DataFrame): DataFrame with election changes
        figures_dir (str): Directory to save figures

    Returns:
        pandas.DataFrame: Recent changes (2016-2020)
    """
    logging.info("Analyzing turnout changes...")

    # Focus on 2016-2020 changes
    recent_changes = election_changes[election_changes['from_year'] == 2016].copy()

    # Calculate national turnout change
    total_votes_2016 = recent_changes['totalvotes_from'].sum()
    total_votes_2020 = recent_changes['totalvotes_to'].sum()
    national_turnout_change = (total_votes_2020 / total_votes_2016 - 1) * 100

    logging.info(f"National turnout change 2016-2020: +{national_turnout_change:.2f}%")

    # Visualize distribution of turnout changes
    plt.figure(figsize=(12, 6))
    sns.histplot(recent_changes['turnout_change_pct'], bins=50, kde=True)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axvline(x=national_turnout_change, color='red', linestyle='--',
                label=f'National Avg: +{national_turnout_change:.2f}%')
    plt.title('Distribution of County Turnout Changes (2016-2020)', fontsize=16)
    plt.xlabel('Change in Turnout (%)', fontsize=14)
    plt.ylabel('Number of Counties', fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'turnout_change_distribution.png'))
    plt.close()

    # Identify highest and lowest turnout change counties
    high_turnout_increase = recent_changes.sort_values('turnout_change_pct', ascending=False).head(20)
    low_turnout_increase = recent_changes.sort_values('turnout_change_pct', ascending=True).head(20)

    logging.info("Top 20 counties with highest turnout increase (2016-2020):")
    for idx, row in high_turnout_increase.iterrows():
        logging.info(f"{row['county_name']}, {row['state_po']}: +{row['turnout_change_pct']:.2f}%")

    logging.info("Top 20 counties with lowest turnout increase/decrease (2016-2020):")
    for idx, row in low_turnout_increase.iterrows():
        logging.info(f"{row['county_name']}, {row['state_po']}: {row['turnout_change_pct']:.2f}%")

    # Analyze relationship between turnout change and partisan shift
    # Create categories for turnout change
    turnout_bins = [-float('inf'), 0, national_turnout_change, 2 * national_turnout_change, float('inf')]
    turnout_labels = ['Decreased', 'Below Average Increase', 'Average Increase', 'Above Average Increase']
    recent_changes['turnout_change_category'] = pd.cut(recent_changes['turnout_change_pct'],
                                                       bins=turnout_bins,
                                                       labels=turnout_labels)

    # Calculate average margin change by turnout category
    turnout_margin_analysis = recent_changes.groupby('turnout_change_category')['margin_change'].agg(['mean', 'count'])
    logging.info("Relationship between turnout change and partisan shift:")
    logging.info(turnout_margin_analysis)

    # Create scatter plot of turnout change vs margin change
    plt.figure(figsize=(12, 8))
    plt.scatter(recent_changes['turnout_change_pct'], recent_changes['margin_change'],
                alpha=0.5, c=recent_changes['margin_from'], cmap=dem_rep_cmap)
    plt.colorbar(label='2016 Margin (+ = Dem, - = Rep)')

    # Add reference lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=national_turnout_change, color='black', linestyle='--', alpha=0.5,
                label=f'National Avg: +{national_turnout_change:.2f}%')

    # Add regression line
    sns.regplot(x='turnout_change_pct', y='margin_change', data=recent_changes,
                scatter=False, color='black')

    plt.title('Turnout Change vs. Partisan Shift (2016-2020)', fontsize=16)
    plt.xlabel('Change in Turnout (%)', fontsize=14)
    plt.ylabel('Change in Margin (+ = Democratic shift, - = Republican shift)', fontsize=14)
    plt.legend()

    plt.savefig(os.path.join(figures_dir, 'turnout_vs_margin_change.png'))
    plt.close()

    # Calculate correlation
    correlation = recent_changes['turnout_change_pct'].corr(recent_changes['margin_change'])
    logging.info(f"Correlation between turnout change and margin change: {correlation:.4f}")

    return recent_changes


def analyze_demographics_and_shifts(election_changes, census_data, merged_data, figures_dir, output_dir):
    """
    Analyze relationships between demographic changes and partisan shifts.

    Args:
        election_changes (pandas.DataFrame): DataFrame with election changes
        census_data (pandas.DataFrame): Census data
        merged_data (pandas.DataFrame): Merged election and census data
        figures_dir (str): Directory to save figures
        output_dir (str): Directory to save output data
    """
    logging.info("Analyzing demographic changes and partisan shifts...")

    # Check if we have data with changes over time
    if 'higher_education_pct_change' in census_data.columns and 'margin_change' in election_changes.columns:
        # Create a merged dataset with both election changes and demographic changes
        counties_2020 = merged_data[merged_data['year'] == 2020].copy()
        counties_2016 = merged_data[merged_data['year'] == 2016].copy()

        # Analyze education and partisan shifts
        if 'higher_education_percentage' in counties_2020.columns:
            plt.figure(figsize=(12, 8))

            # Use county_fips instead of fips for election_changes
            edu_margin_data = pd.merge(
                counties_2020[['fips', 'higher_education_percentage']],
                election_changes[['county_fips', 'margin_change']],
                left_on='fips',
                right_on='county_fips',
                how='inner'
            )

            plt.scatter(edu_margin_data['higher_education_percentage'],
                        edu_margin_data['margin_change'],
                        alpha=0.5, c=edu_margin_data['margin_change'], cmap=dem_rep_cmap)
            plt.colorbar(label='Margin Change (+ = Dem shift, - = Rep shift)')

            # Add reference line
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            # Add regression line
            sns.regplot(x='higher_education_percentage', y='margin_change',
                        data=edu_margin_data,
                        scatter=False, color='black')

            plt.title('Higher Education and Partisan Shifts (2016-2020)', fontsize=16)
            plt.xlabel('Higher Education Percentage (2020)', fontsize=14)
            plt.ylabel('Change in Margin (+ = Democratic shift, - = Republican shift)', fontsize=14)

            plt.savefig(os.path.join(figures_dir, 'education_vs_partisan_shift.png'))
            plt.close()

            # Calculate correlation
            edu_shift_corr = edu_margin_data['higher_education_percentage'].corr(edu_margin_data['margin_change'])
            logging.info(f"Correlation between higher education % and partisan shift: {edu_shift_corr:.4f}")

        # Analyze unemployment and partisan shifts
        if 'unemployment_rate' in counties_2020.columns and 'unemployment_rate' in counties_2016.columns:
            # Calculate unemployment rate change
            unemployment_change = pd.merge(
                counties_2016[['fips', 'unemployment_rate']],
                counties_2020[['fips', 'unemployment_rate']],
                on='fips',
                suffixes=('_2016', '_2020')
            )
            unemployment_change['unemployment_change'] = unemployment_change['unemployment_rate_2020'] - \
                                                         unemployment_change['unemployment_rate_2016']

            # Merge with partisan shifts - note the use of county_fips instead of fips
            unemp_shift_data = pd.merge(
                unemployment_change,
                election_changes[['county_fips', 'margin_change']],
                left_on='fips',
                right_on='county_fips',
                how='inner'
            )

            # Visualize relationship
            plt.figure(figsize=(12, 8))
            plt.scatter(unemp_shift_data['unemployment_change'],
                        unemp_shift_data['margin_change'],
                        alpha=0.5)

            # Add reference lines
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            # Add regression line
            sns.regplot(x='unemployment_change', y='margin_change',
                        data=unemp_shift_data,
                        scatter=False, color='black')

            plt.title('Unemployment Change and Partisan Shifts (2016-2020)', fontsize=16)
            plt.xlabel('Change in Unemployment Rate (2016-2020)', fontsize=14)
            plt.ylabel('Change in Margin (+ = Democratic shift, - = Republican shift)', fontsize=14)

            plt.savefig(os.path.join(figures_dir, 'unemployment_change_vs_partisan_shift.png'))
            plt.close()

            # Calculate correlation
            unemp_shift_corr = unemp_shift_data['unemployment_change'].corr(unemp_shift_data['margin_change'])
            logging.info(f"Correlation between unemployment change and partisan shift: {unemp_shift_corr:.4f}")

            # Categorize counties by unemployment change
            unemp_shift_data['unemployment_change_cat'] = pd.cut(
                unemp_shift_data['unemployment_change'],
                bins=[-float('inf'), -1, 0, 1, float('inf')],
                labels=['Improved >1%', 'Improved 0-1%', 'Worsened 0-1%', 'Worsened >1%']
            )

            # Calculate average margin change by unemployment change category
            unemp_margin_analysis = unemp_shift_data.groupby('unemployment_change_cat')['margin_change'].agg(
                ['mean', 'count'])
            logging.info("Relationship between unemployment change and partisan shift:")
            logging.info(unemp_margin_analysis)
    else:
        logging.warning("Insufficient data to analyze demographic changes over time")

    # Analyze competitiveness and education
    counties_2020 = merged_data[merged_data['year'] == 2020].copy()

    # Define competitive counties (margin within ±5 points)
    counties_2020['competitiveness'] = pd.cut(
        counties_2020['margin'].abs(),
        bins=[0, 5, 15, float('inf')],
        labels=['Highly Competitive (±5%)', 'Competitive (±5-15%)', 'Safe (>15%)']
    )

    # Analyze average education by competitiveness
    if 'higher_education_percentage' in counties_2020.columns:
        edu_by_comp = counties_2020.groupby('competitiveness')['higher_education_percentage'].agg(['mean', 'count'])
        logging.info("Relationship between competitiveness and education:")
        logging.info(edu_by_comp)

        # Visualize
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='competitiveness', y='higher_education_percentage', data=counties_2020)
        plt.title('Education Levels by County Competitiveness (2020)', fontsize=16)
        plt.xlabel('Competitiveness Category', fontsize=14)
        plt.ylabel('Higher Education Percentage', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(os.path.join(figures_dir, 'education_by_competitiveness.png'))
        plt.close()

    # Check what columns are available before saving to CSV
    available_columns = counties_2020.columns.tolist()
    csv_columns = []

    # Only include columns that exist in the DataFrame
    for col in ['county_name', 'state_po', 'competitiveness', 'margin', 'higher_education_percentage']:
        if col in available_columns:
            csv_columns.append(col)

    if csv_columns:  # Only save if we have columns to save
        counties_2020[csv_columns].sort_values(['competitiveness', 'margin']) \
            .to_csv(os.path.join(output_dir, 'county_competitiveness_2020.csv'), index=False)
        logging.info(f"Saved county competitiveness analysis with columns: {csv_columns}")
    else:
        logging.warning("Could not save county competitiveness CSV due to missing columns")

    return counties_2020

def analyze_rural_urban_divide(merged_data, election_changes, figures_dir, output_dir):
    """
    Analyze the rural-urban divide in voting patterns and shifts.

    Args:
        merged_data (pandas.DataFrame): Merged election and census data
        election_changes (pandas.DataFrame): DataFrame with election changes
        figures_dir (str): Directory to save figures
        output_dir (str): Directory to save output data

    Returns:
        pandas.DataFrame: Changes with density categorization
    """
    logging.info("Analyzing rural-urban divide...")

    # Check if we have rural/urban classification or population density
    if 'population_density' in merged_data.columns:
        # Focus on 2020 data
        counties_2020 = merged_data[merged_data['year'] == 2020].copy()

        # Create population density categories
        counties_2020['density_category'] = pd.qcut(
            counties_2020['population_density'],
            q=5,  # quintiles
            labels=['Very Rural', 'Rural', 'Suburban', 'Urban', 'Very Urban']
        )

        # Analyze partisan lean by density category
        density_analysis = counties_2020.groupby('density_category')['margin'].agg(['mean', 'median', 'count'])
        logging.info("Partisan lean by population density (2020):")
        logging.info(density_analysis)

        # Visualize partisan lean by density
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='density_category', y='margin', data=counties_2020)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Partisan Lean by Population Density (2020)', fontsize=16)
        plt.xlabel('Population Density Category', fontsize=14)
        plt.ylabel('Partisan Margin (+ = Democratic, - = Republican)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'partisan_lean_by_density.png'))
        plt.close()

        # Now analyze shifts by density category
        # First, merge density categories with election changes
        recent_changes = election_changes[election_changes['from_year'] == 2016].copy()
        county_density = counties_2020[['fips', 'density_category']].copy()

        # Fix: Use county_fips instead of fips for the merge
        changes_with_density = pd.merge(
            recent_changes,
            county_density,
            left_on='county_fips',  # Changed from fips to county_fips
            right_on='fips',
            how='inner'
        )

        # Analyze margin changes by density category
        shift_analysis = changes_with_density.groupby('density_category')['margin_change'].agg(
            ['mean', 'median', 'count'])
        logging.info("Partisan shifts by population density (2016-2020):")
        logging.info(shift_analysis)

        # Visualize shifts by density
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='density_category', y='margin_change', data=changes_with_density)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Partisan Shifts by Population Density (2016-2020)', fontsize=16)
        plt.xlabel('Population Density Category', fontsize=14)
        plt.ylabel('Change in Margin (+ = Democratic shift, - = Republican shift)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'partisan_shift_by_density.png'))
        plt.close()

        # Save analysis to CSV
        changes_with_density.to_csv(os.path.join(output_dir, 'partisan_shifts_by_density.csv'), index=False)

        return changes_with_density
    else:
        logging.warning("Population density data not available for rural-urban analysis")
        return None


def predict_voter_turnout(merged_data, election_changes, figures_dir, output_dir):
    """
    Analyze factors that predict voter turnout and changes in turnout.

    Args:
        merged_data (pandas.DataFrame): Merged election and census data
        election_changes (pandas.DataFrame): DataFrame with election changes
        figures_dir (str): Directory to save figures
        output_dir (str): Directory to save output data

    Returns:
        pandas.DataFrame: 2020 counties data with turnout analysis
    """
    logging.info("Analyzing predictors of voter turnout...")

    # Focus on 2020 data
    counties_2020 = merged_data[merged_data['year'] == 2020].copy()

    # Log the available columns to help with debugging
    available_columns = counties_2020.columns.tolist()
    logging.info(f"Available columns in counties_2020: {available_columns}")

    # Calculate turnout percentage if we have population data
    if 'total_population' in counties_2020.columns:
        counties_2020['turnout_percentage'] = counties_2020['totalvotes'] / counties_2020['total_population'] * 100
        logging.info(f"Average county turnout percentage in 2020: {counties_2020['turnout_percentage'].mean():.2f}%")

        # Check for required columns before accessing them
        required_columns = ['county_name', 'state_po', 'turnout_percentage']
        if all(col in counties_2020.columns for col in required_columns):
            # Identify highest and lowest turnout counties
            high_turnout = counties_2020.sort_values('turnout_percentage', ascending=False).head(20)
            low_turnout = counties_2020.sort_values('turnout_percentage', ascending=True).head(20)

            logging.info("Top 20 counties with highest turnout percentage (2020):")
            for idx, row in high_turnout.iterrows():
                logging.info(f"{row['county_name']}, {row['state_po']}: {row['turnout_percentage']:.2f}%")

            logging.info("Top 20 counties with lowest turnout percentage (2020):")
            for idx, row in low_turnout.iterrows():
                logging.info(f"{row['county_name']}, {row['state_po']}: {row['turnout_percentage']:.2f}%")
        else:
            missing = [col for col in required_columns if col not in counties_2020.columns]
            logging.warning(f"Missing required columns for detailed turnout analysis: {missing}")
            logging.info("Skipping detailed county turnout analysis due to missing columns")

        # Analyze relationship between competitiveness and turnout
        counties_2020['competitiveness'] = pd.cut(
            counties_2020['margin'].abs(),
            bins=[0, 5, 15, float('inf')],
            labels=['Highly Competitive (±5%)', 'Competitive (±5-15%)', 'Safe (>15%)']
        )

        turnout_by_comp = counties_2020.groupby('competitiveness')['turnout_percentage'].agg(
            ['mean', 'median', 'count'])
        logging.info("Turnout by competitiveness (2020):")
        logging.info(turnout_by_comp)

        # Visualize relationship
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='competitiveness', y='turnout_percentage', data=counties_2020)
        plt.title('Voter Turnout by County Competitiveness (2020)', fontsize=16)
        plt.xlabel('Competitiveness Category', fontsize=14)
        plt.ylabel('Turnout Percentage', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'turnout_by_competitiveness.png'))
        plt.close()

        # Analyze relationship between education and turnout if both columns exist
        if 'higher_education_percentage' in counties_2020.columns:
            plt.figure(figsize=(12, 8))
            plt.scatter(counties_2020['higher_education_percentage'],
                        counties_2020['turnout_percentage'],
                        alpha=0.5, c=counties_2020['margin'], cmap=dem_rep_cmap)
            plt.colorbar(label='2020 Margin (+ = Dem, - = Rep)')

            # Add regression line
            sns.regplot(x='higher_education_percentage', y='turnout_percentage',
                        data=counties_2020,
                        scatter=False, color='black')

            plt.title('Higher Education and Voter Turnout (2020)', fontsize=16)
            plt.xlabel('Higher Education Percentage', fontsize=14)
            plt.ylabel('Turnout Percentage', fontsize=14)

            plt.savefig(os.path.join(figures_dir, 'education_vs_turnout.png'))
            plt.close()

            # Calculate correlation
            edu_turnout_corr = counties_2020['higher_education_percentage'].corr(counties_2020['turnout_percentage'])
            logging.info(f"Correlation between higher education % and turnout: {edu_turnout_corr:.4f}")

        # Look at changes in turnout from 2016 to 2020
        recent_changes = election_changes[election_changes['from_year'] == 2016].copy()

        # Analyze relationship between income and turnout change
        if 'median_household_income' in counties_2020.columns:
            # Fix: Use county_fips instead of fips for the merge
            income_turnout = pd.merge(
                counties_2020[['fips', 'median_household_income']],
                recent_changes[['county_fips', 'turnout_change_pct']],
                left_on='fips',
                right_on='county_fips',
                how='inner'
            )

            plt.figure(figsize=(12, 8))
            plt.scatter(income_turnout['median_household_income'],
                        income_turnout['turnout_change_pct'],
                        alpha=0.5)

            # Add regression line
            sns.regplot(x='median_household_income', y='turnout_change_pct',
                        data=income_turnout,
                        scatter=False, color='black')

            plt.title('Household Income and Turnout Change (2016-2020)', fontsize=16)
            plt.xlabel('Median Household Income (2020)', fontsize=14)
            plt.ylabel('Change in Turnout (%)', fontsize=14)

            plt.savefig(os.path.join(figures_dir, 'income_vs_turnout_change.png'))
            plt.close()

            # Calculate correlation
            income_turnout_corr = income_turnout['median_household_income'].corr(income_turnout['turnout_change_pct'])
            logging.info(f"Correlation between household income and turnout change: {income_turnout_corr:.4f}")

        # Save turnout analysis to CSV - check for required columns first
        csv_columns = []
        for col in ['county_name', 'state_po', 'turnout_percentage', 'competitiveness', 'higher_education_percentage']:
            if col in counties_2020.columns:
                csv_columns.append(col)

        if csv_columns:
            # Only sort by columns that exist
            sort_columns = [col for col in ['turnout_percentage'] if col in csv_columns]
            if sort_columns:
                counties_2020[csv_columns].sort_values(sort_columns, ascending=False) \
                    .to_csv(os.path.join(output_dir, 'county_turnout_analysis_2020.csv'), index=False)
            else:
                counties_2020[csv_columns].to_csv(os.path.join(output_dir, 'county_turnout_analysis_2020.csv'),
                                                  index=False)

            logging.info(f"Saved county turnout analysis with columns: {csv_columns}")
        else:
            logging.warning("Could not save county turnout CSV due to missing columns")

        return counties_2020
    else:
        logging.warning("Population data not available for turnout percentage calculation")
        return None