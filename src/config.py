#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration module for political trends analysis.
Contains setup functions for directories and logging.
"""

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def setup_directories():
    """
    Set up directories needed for the analysis.

    Returns:
        dict: Dictionary with directory paths
    """
    # Set up paths and directories
    print("Setting up directories...")
    data_dir = 'data/raw'
    output_dir = 'data/analysis'
    figures_dir = 'reports/figures'

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    return {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'figures_dir': figures_dir
    }


def setup_logging(log_dir):
    """
    Configure logging for the application.

    Args:
        log_dir (str): Directory for log files
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    log_file = os.path.join(log_dir, 'analysis.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )


def setup_plotting():
    """
    Configure matplotlib and seaborn for consistent visualizations.

    Returns:
        dict: Dictionary with plotting configuration including color palettes
    """
    # Set plot style - using updated style name for newer matplotlib versions
    try:
        plt.style.use('seaborn-v0_8-whitegrid')  # For newer matplotlib
    except:
        try:
            plt.style.use('seaborn-whitegrid')  # For older matplotlib
        except:
            print("Warning: Could not set seaborn style. Using default style.")

    sns.set_palette('viridis')
    plt.rcParams['figure.figsize'] = [12, 8]

    # Define custom color palettes
    dem_rep_cmap = LinearSegmentedColormap.from_list('dem_rep', ['blue', 'white', 'red'])

    return {
        'dem_rep_cmap': dem_rep_cmap
    }