#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization module for political trends analysis.
Contains functions for creating various visualizations.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from config import setup_plotting

# Get plotting configuration
plot_config = setup_plotting()
dem_rep_cmap = plot_config['dem_rep_cmap']


def create_margin_map(county_data, state_boundaries=None, year=2020, output_dir=None):
    """
    Create a county-level map of election margins.

    Args:
        county_data (pandas.DataFrame): County election data
        state_boundaries (geopandas.GeoDataFrame, optional): State boundary data
        year (int, optional): Election year to visualize
        output_dir (str, optional): Directory to save output files

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    try:
        import geopandas as gpd
        logging.info("Creating county-level margin map...")

        # Filter to specified year
        data = county_data[county_data['year'] == year].copy()

        # Load county boundaries if not provided
        counties_map = None
        try:
            # Try to load from standard locations
            try:
                counties_map = gpd.read_file('data/geo/counties.shp')
            except:
                counties_map = gpd.read_file(
                    'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip')

            # Ensure FIPS codes are formatted consistently
            counties_map['GEOID'] = counties_map['GEOID'].astype(str).str.zfill(5)

            # Merge election data with geographic data
            merged_map = counties_map.merge(
                data,
                left_on='GEOID',
                right_on='county_fips',
                how='left'
            )

            # Create the map
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            # Set up color scheme (blue for Democratic, red for Republican)
            vmin = -30
            vmax = 30
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

            # Plot counties
            merged_map.plot(
                column='margin',
                cmap=dem_rep_cmap,
                norm=norm,
                linewidth=0.1,
                edgecolor='0.5',
                ax=ax,
                legend=True,
                missing_kwds={'color': 'lightgray'}
            )

            # Add state boundaries if provided
            if state_boundaries is not None:
                state_boundaries.boundary.plot(ax=ax, linewidth=0.5, color='black')

            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=dem_rep_cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Democratic vs. Republican Margin (percentage points)')

            # Add title and adjust layout
            plt.title(f'County-Level Presidential Election Results ({year})', fontsize=16)
            plt.tight_layout()

            # Save figure if output directory provided
            if output_dir:
                output_file = os.path.join(output_dir, f'county_margin_map_{year}.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logging.info(f"Map saved to {output_file}")

            return fig

        except Exception as e:
            logging.error(f"Error creating map: {str(e)}")
            return None

    except ImportError:
        logging.warning("GeoPandas not installed. Cannot create geographic visualizations.")
        return None


def create_margin_change_map(election_changes, state_boundaries=None, from_year=2016, to_year=2020, output_dir=None):
    """
    Create a county-level map of election margin changes.

    Args:
        election_changes (pandas.DataFrame): County election change data
        state_boundaries (geopandas.GeoDataFrame, optional): State boundary data
        from_year (int, optional): Starting year for change calculation
        to_year (int, optional): Ending year for change calculation
        output_dir (str, optional): Directory to save output files

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    try:
        import geopandas as gpd
        logging.info(f"Creating county-level margin change map ({from_year}-{to_year})...")

        # Filter to specified years
        data = election_changes[(election_changes['from_year'] == from_year) &
                                (election_changes['to_year'] == to_year)].copy()

        # Load county boundaries if not provided
        counties_map = None
        try:
            # Try to load from standard locations
            try:
                counties_map = gpd.read_file('data/geo/counties.shp')
            except:
                counties_map = gpd.read_file(
                    'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip')

            # Ensure FIPS codes are formatted consistently
            counties_map['GEOID'] = counties_map['GEOID'].astype(str).str.zfill(5)

            # Merge election data with geographic data
            merged_map = counties_map.merge(
                data,
                left_on='GEOID',
                right_on='county_fips',
                how='left'
            )

            # Create the map
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            # Set up color scheme (blue for Democratic shift, red for Republican shift)
            vmin = -15
            vmax = 15
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

            # Plot counties
            merged_map.plot(
                column='margin_change',
                cmap=dem_rep_cmap,
                norm=norm,
                linewidth=0.1,
                edgecolor='0.5',
                ax=ax,
                legend=True,
                missing_kwds={'color': 'lightgray'}
            )

            # Add state boundaries if provided
            if state_boundaries is not None:
                state_boundaries.boundary.plot(ax=ax, linewidth=0.5, color='black')

            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=dem_rep_cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Change in Democratic vs. Republican Margin (percentage points)')

            # Add title and adjust layout
            plt.title(f'County-Level Election Shifts ({from_year}-{to_year})', fontsize=16)
            plt.tight_layout()

            # Save figure if output directory provided
            if output_dir:
                output_file = os.path.join(output_dir, f'county_margin_change_map_{from_year}_{to_year}.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logging.info(f"Map saved to {output_file}")

            return fig

        except Exception as e:
            logging.error(f"Error creating map: {str(e)}")
            return None

    except ImportError:
        logging.warning("GeoPandas not installed. Cannot create geographic visualizations.")
        return None


def create_turnout_change_map(election_changes, state_boundaries=None, from_year=2016, to_year=2020, output_dir=None):
    """
    Create a county-level map of voter turnout changes.

    Args:
        election_changes (pandas.DataFrame): County election change data
        state_boundaries (geopandas.GeoDataFrame, optional): State boundary data
        from_year (int, optional): Starting year for change calculation
        to_year (int, optional): Ending year for change calculation
        output_dir (str, optional): Directory to save output files

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    try:
        import geopandas as gpd
        logging.info(f"Creating county-level turnout change map ({from_year}-{to_year})...")

        # Filter to specified years
        data = election_changes[(election_changes['from_year'] == from_year) &
                                (election_changes['to_year'] == to_year)].copy()

        # Load county boundaries if not provided
        counties_map = None
        try:
            # Try to load from standard locations
            try:
                counties_map = gpd.read_file('data/geo/counties.shp')
            except:
                counties_map = gpd.read_file(
                    'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip')

            # Ensure FIPS codes are formatted consistently
            counties_map['GEOID'] = counties_map['GEOID'].astype(str).str.zfill(5)

            # Merge election data with geographic data
            merged_map = counties_map.merge(
                data,
                left_on='GEOID',
                right_on='county_fips',
                how='left'
            )

            # Create the map
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            # Calculate national average turnout change for reference
            avg_turnout_change = data['turnout_change_pct'].mean()

            # Set up color scheme (darker green for higher turnout increase)
            vmin = -10
            vmax = 40
            vcenter = avg_turnout_change
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

            # Create a custom colormap for turnout (red for decrease, white for average, green for increase)
            turnout_cmap = LinearSegmentedColormap.from_list('turnout', ['red', 'white', 'green'])

            # Plot counties
            merged_map.plot(
                column='turnout_change_pct',
                cmap=turnout_cmap,
                norm=norm,
                linewidth=0.1,
                edgecolor='0.5',
                ax=ax,
                legend=True,
                missing_kwds={'color': 'lightgray'}
            )

            # Add state boundaries if provided
            if state_boundaries is not None:
                state_boundaries.boundary.plot(ax=ax, linewidth=0.5, color='black')

            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=turnout_cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Change in Voter Turnout (%)')

            # Add reference line for national average
            cbar.ax.axhline(y=norm(avg_turnout_change), color='black', linestyle='--', linewidth=1)
            cbar.ax.text(1.5, norm(avg_turnout_change), f' Nat. Avg: {avg_turnout_change:.1f}%',
                         va='center', ha='left', fontsize=8, color='black')

            # Add title and adjust layout
            plt.title(f'County-Level Voter Turnout Changes ({from_year}-{to_year})', fontsize=16)
            plt.tight_layout()

            # Save figure if output directory provided
            if output_dir:
                output_file = os.path.join(output_dir, f'county_turnout_change_map_{from_year}_{to_year}.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logging.info(f"Map saved to {output_file}")

            return fig

        except Exception as e:
            logging.error(f"Error creating map: {str(e)}")
            return None

    except ImportError:
        logging.warning("GeoPandas not installed. Cannot create geographic visualizations.")
        return None


def create_demographic_correlation_plot(merged_data, variable, year=2020, output_dir=None):
    """
    Create a scatter plot showing correlation between a demographic variable and election results.

    Args:
        merged_data (pandas.DataFrame): Merged election and census data
        variable (str): Census variable to analyze (e.g., 'higher_education_percentage')
        year (int, optional): Election year to visualize
        output_dir (str, optional): Directory to save output files

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logging.info(f"Creating demographic correlation plot for {variable}...")

    # Filter to specified year
    data = merged_data[merged_data['year'] == year].copy()

    # Check if variable exists in data
    if variable not in data.columns:
        logging.warning(f"Variable '{variable}' not found in data")
        return None

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine variable label
    variable_labels = {
        'higher_education_percentage': 'Higher Education (%)',
        'median_household_income': 'Median Household Income ($)',
        'unemployment_rate': 'Unemployment Rate (%)',
        'population_density': 'Population Density (people/sq mi)',
        'poverty_rate': 'Poverty Rate (%)'
    }

    variable_label = variable_labels.get(variable, variable)

    # Create the scatter plot, color by margin
    scatter = ax.scatter(
        data[variable],
        data['margin'],
        alpha=0.5,
        c=data['margin'],
        cmap=dem_rep_cmap,
        s=data['totalvotes'] / 10000  # Size by total votes
    )

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Democratic vs. Republican Margin (percentage points)')

    # Add reference line at y=0 (tied election)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Add regression line
    sns.regplot(x=variable, y='margin', data=data, scatter=False, color='black', ax=ax)

    # Calculate correlation
    correlation = data[variable].corr(data['margin'])

    # Add titles and labels
    ax.set_title(f'Relationship between {variable_label} and Election Results ({year})', fontsize=16)
    ax.set_xlabel(variable_label, fontsize=14)
    ax.set_ylabel('Democratic vs. Republican Margin (percentage points)', fontsize=14)

    # Add correlation annotation
    ax.annotate(f'Correlation: {correlation:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Add a legend for the size of points
    sizes = [10000, 50000, 100000, 500000]
    labels = ['10k votes', '50k votes', '100k votes', '500k votes']
    legend_points = [plt.scatter([], [], s=size / 10000, c='gray', alpha=0.5) for size in sizes]
    plt.legend(legend_points, labels, scatterpoints=1, title='Total Votes',
               loc='lower right', frameon=True)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output directory provided
    if output_dir:
        output_file = os.path.join(output_dir, f'{variable}_vs_margin_{year}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {output_file}")

    return fig