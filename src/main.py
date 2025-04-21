#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Political Trends and Demographic Insights Analysis

Main entry point script that coordinates the entire analysis workflow.
"""

import logging
from src.data_collection.data_loading import load_census_data, load_election_data, merge_data
from analysis import (
    analyze_partisan_shifts,
    analyze_turnout_changes,
    analyze_demographics_and_shifts,
    analyze_rural_urban_divide,
    predict_voter_turnout
)
from reporting import generate_summary_report
from config import setup_directories, setup_logging


def main():
    """
    Main function to run the analysis.
    """
    # Setup directories and logging
    config = setup_directories()
    setup_logging(config['output_dir'])

    logging.info("Political Trends and Demographic Insights Analysis")
    logging.info("=================================================")

    try:
        # Load data
        logging.info("Loading data...")
        census_data = load_census_data(config['data_dir'])
        county_results, election_changes = load_election_data(config['data_dir'])

        # Merge datasets
        if 'fips' in census_data.columns:
            merged_data = merge_data(county_results, census_data)
        else:
            logging.warning("Cannot merge election and census data due to missing FIPS codes")
            merged_data = county_results

        # Run analyses - with proper error handling for each function
        logging.info("Running analyses...")

        try:
            recent_changes = analyze_partisan_shifts(election_changes, config['figures_dir'], config['output_dir'])
        except Exception as e:
            logging.error(f"Error in partisan shifts analysis: {str(e)}")
            recent_changes = election_changes[election_changes['from_year'] == 2016].copy()

        try:
            turnout_analysis = analyze_turnout_changes(election_changes, config['figures_dir'])
        except Exception as e:
            logging.error(f"Error in turnout changes analysis: {str(e)}")

        try:
            analyze_demographics_and_shifts(election_changes, census_data, merged_data, config['figures_dir'],
                                            config['output_dir'])
        except Exception as e:
            logging.error(f"Error in demographics and shifts analysis: {str(e)}")

        try:
            rural_urban_analysis = analyze_rural_urban_divide(merged_data, election_changes, config['figures_dir'],
                                                              config['output_dir'])
        except Exception as e:
            logging.error(f"Error in rural-urban divide analysis: {str(e)}")

        try:
            counties_2020 = predict_voter_turnout(merged_data, election_changes, config['figures_dir'],
                                                  config['output_dir'])
        except Exception as e:
            logging.error(f"Error in voter turnout prediction: {str(e)}")
            counties_2020 = merged_data[merged_data['year'] == 2020].copy() if 'year' in merged_data.columns else None

        # Generate summary report - only if we have the necessary data
        if recent_changes is not None and merged_data is not None and counties_2020 is not None:
            try:
                logging.info("Generating summary report...")
                generate_summary_report(recent_changes, merged_data, counties_2020, config['output_dir'])
            except Exception as e:
                logging.error(f"Error generating summary report: {str(e)}")
        else:
            logging.warning("Skipping summary report generation due to missing analysis data")

        logging.info("Analysis complete!")
        logging.info(f"Results saved to {config['output_dir']}")
        logging.info(f"Visualizations saved to {config['figures_dir']}")

    except Exception as e:
        logging.error(f"ERROR: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()