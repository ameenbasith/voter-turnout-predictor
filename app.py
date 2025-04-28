import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Education & Voting Patterns Analyzer",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎓 Education & Voting Patterns Analyzer 🗳️</div>', unsafe_allow_html=True)
st.markdown("""
This interactive dashboard analyzes the relationship between education levels and political leanings 
at both state and county levels, using data from the MIT Election Lab and U.S. Census Bureau.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "State Analysis", "County Shifts", "About"]
)

# EXACT file paths based on your project structure
DATA_PATHS = {
    'analysis': 'src/data/analysis/',
    'processed': 'src/data/processed/',
    'raw': 'src/data/raw/'
}

# Specific file paths
FILE_PATHS = {
    'flipped_dem_to_rep': os.path.join(DATA_PATHS['analysis'], 'flipped_dem_to_rep_2016_2020.csv'),
    'flipped_rep_to_dem': os.path.join(DATA_PATHS['analysis'], 'flipped_rep_to_dem_2016_2020.csv'),
    'national_turnout': os.path.join(DATA_PATHS['analysis'], 'national_turnout_by_year.csv'),
    'partisan_lean': os.path.join(DATA_PATHS['analysis'], 'partisan_lean_by_education_quintile.csv'),
    'top_dem_shifts': os.path.join(DATA_PATHS['analysis'], 'top_dem_shifts.csv'),
    'top_rep_shifts': os.path.join(DATA_PATHS['analysis'], 'top_rep_shifts.csv'),
    'state_education': os.path.join(DATA_PATHS['processed'], 'state_education_summary.csv'),
    'state_pivot': os.path.join(DATA_PATHS['processed'], 'state_pivot_summary.csv'),
    'census_data': os.path.join(DATA_PATHS['raw'], 'census_county_data_all_years.csv'),
    'election_data': os.path.join(DATA_PATHS['raw'], 'mit_election_lab_county_returns_raw.csv')
}


# Function to load data with better error handling
def load_file(file_key, file_path):
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            st.sidebar.success(f"✅ Loaded: {file_key}")
            return df
        else:
            st.sidebar.error(f"❌ File not found: {file_path}")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {file_key}: {str(e)}")
        return None


# Function to load all data
@st.cache_data
def load_data():
    data = {}
    for key, path in FILE_PATHS.items():
        data[key] = load_file(key, path)
    return data


# Load data
data = load_data()

# Check if we have the minimum required data
has_any_data = any(value is not None for value in data.values())
if not has_any_data:
    st.error("No data files could be loaded. Please check your file paths.")

    st.info("""
    ## Troubleshooting File Paths

    Based on your project structure, the app is looking for files in:

    - Analysis data: `src/data/analysis/`
    - Processed data: `src/data/processed/`
    - Raw data: `src/data/raw/`

    Please make sure these directories exist and contain the expected files. 
    Here's what's being looked for in each directory:

    **src/data/analysis/**
    - flipped_dem_to_rep_2016_2020.csv
    - flipped_rep_to_dem_2016_2020.csv
    - national_turnout_by_year.csv
    - partisan_lean_by_education_quintile.csv
    - top_dem_shifts.csv
    - top_rep_shifts.csv

    **src/data/processed/**
    - state_education_summary.csv
    - state_pivot_summary.csv

    **src/data/raw/**
    - census_county_data_all_years.csv
    - mit_election_lab_county_returns_raw.csv
    """)

    # Show the current directory for debugging
    st.code(f"Current working directory: {os.getcwd()}")

    st.stop()

# Overview Page
if page == "Overview":
    st.markdown('<div class="sub-header">National Election Overview</div>', unsafe_allow_html=True)

    # Section 1: Flipped Counties Summary
    st.markdown("### County Shifts Between 2016-2020")

    col1, col2 = st.columns(2)

    with col1:
        rep_to_dem_count = len(data['flipped_rep_to_dem']) if data['flipped_rep_to_dem'] is not None else 0
        dem_to_rep_count = len(data['flipped_dem_to_rep']) if data['flipped_dem_to_rep'] is not None else 0

        # Create a simple bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(['Rep → Dem', 'Dem → Rep'], [rep_to_dem_count, dem_to_rep_count],
               color=['blue', 'red'])
        ax.set_title('Counties that Flipped Between 2016-2020')
        ax.set_ylabel('Number of Counties')

        # Add count labels on top of bars
        for i, v in enumerate([rep_to_dem_count, dem_to_rep_count]):
            ax.text(i, v + 0.5, str(v), ha='center')

        st.pyplot(fig)

    with col2:
        # Get top shifts
        if data['top_dem_shifts'] is not None and data['top_rep_shifts'] is not None:
            top_dem = data['top_dem_shifts'].head(5)
            top_rep = data['top_rep_shifts'].head(5)

            st.markdown("### Top Democratic Shifts")
            st.dataframe(top_dem)

            st.markdown("### Top Republican Shifts")
            st.dataframe(top_rep)

    # Section 2: National Turnout Trend
    st.markdown("### Voter Turnout Trend")

    if data['national_turnout'] is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['national_turnout']['year'], data['national_turnout']['national_turnout_pct'],
                marker='o', linestyle='-', linewidth=2)
        ax.set_title('National Turnout Percentage by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Turnout (%)')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add percent labels on points
        for i, v in enumerate(data['national_turnout']['national_turnout_pct']):
            ax.text(data['national_turnout']['year'].iloc[i], v + 0.5, f"{v:.1f}%", ha='center')

        st.pyplot(fig)
    else:
        st.warning("National turnout data not available.")

    # Section 3: Education and Partisan Lean
    st.markdown("### Education Levels and Partisan Lean")

    if data['partisan_lean'] is not None:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot both mean and median
        x = data['partisan_lean']['education_quintile']
        mean_vals = data['partisan_lean']['mean']
        median_vals = data['partisan_lean']['median']

        x_pos = np.arange(len(x))
        width = 0.35

        ax.bar(x_pos - width / 2, mean_vals, width, label='Mean', color='skyblue')
        ax.bar(x_pos + width / 2, median_vals, width, label='Median', color='orange')

        ax.set_title('Partisan Lean by Education Quintile')
        ax.set_ylabel('Partisan Lean')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')

        st.pyplot(fig)
    else:
        st.warning("Partisan lean data not available.")

# State Analysis Page
elif page == "State Analysis":
    st.markdown('<div class="sub-header">State-Level Analysis</div>', unsafe_allow_html=True)

    # Check if we have state pivot and education data
    if data['state_pivot'] is not None and data['state_education'] is not None:
        # Get available years
        years = sorted(data['state_pivot']['year'].unique())
        selected_year = st.selectbox("Select Year", options=years, index=len(years) - 1)

        # Filter data for selected year
        state_pivot_year = data['state_pivot'][data['state_pivot']['year'] == selected_year]
        state_edu_year = data['state_education'][data['state_education']['year'] == selected_year]

        # Create state pivot table
        st.markdown("### State Vote Margins")
        st.dataframe(state_pivot_year.sort_values('dem_margin', ascending=False))

        # Attempt to join data for scatterplot
        try:
            # Merge datasets
            state_merged = pd.merge(
                state_pivot_year,
                state_edu_year,
                left_on=['state_po'],
                right_on=['state'],
                how='inner'
            )

            if len(state_merged) > 0:
                # Calculate higher education percentage
                edu_cols = [col for col in state_merged.columns if 'degree_pct' in col]
                if edu_cols:
                    state_merged['higher_education'] = state_merged[edu_cols].sum(axis=1)

                    # Create scatterplot
                    st.markdown("### Education vs. Democratic Margin")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(
                        state_merged['higher_education'],
                        state_merged['dem_margin'],
                        c=state_merged['dem_margin'],
                        cmap='RdBu',
                        s=100,
                        alpha=0.7
                    )

                    # Add state labels
                    for i, row in state_merged.iterrows():
                        ax.annotate(row['state_po'],
                                    (row['higher_education'], row['dem_margin']),
                                    xytext=(5, 5),
                                    textcoords='offset points')

                    # Add correlation line
                    z = np.polyfit(state_merged['higher_education'], state_merged['dem_margin'], 1)
                    p = np.poly1d(z)
                    ax.plot(state_merged['higher_education'], p(state_merged['higher_education']), "r--", alpha=0.8)

                    # Add correlation coefficient
                    corr = state_merged['higher_education'].corr(state_merged['dem_margin'])
                    ax.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax.transAxes,
                            fontsize=12, verticalalignment='top')

                    # Styling
                    ax.set_xlabel('Higher Education Percentage')
                    ax.set_ylabel('Democratic Margin')
                    ax.set_title(f'Relationship Between Education and Democratic Support ({selected_year})')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

                    # Add colorbar
                    plt.colorbar(scatter, label='Democratic Margin')

                    st.pyplot(fig)
                else:
                    st.warning("Education columns not found in the data.")
            else:
                st.warning("No matching states between pivot and education data.")
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
    else:
        st.warning("State pivot or education data not available.")

# County Shifts Page
elif page == "County Shifts":
    st.markdown('<div class="sub-header">County-Level Shifts</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Counties that Flipped Democrat to Republican")
        if data['flipped_dem_to_rep'] is not None:
            # Group by state and count
            state_counts = data['flipped_dem_to_rep']['state_po'].value_counts().reset_index()
            state_counts.columns = ['State', 'Count']

            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, len(state_counts) * 0.4 + 2))
            ax.barh(state_counts['State'], state_counts['Count'], color='red')
            ax.set_xlabel('Number of Counties')
            ax.set_title('Counties Flipped Dem → Rep by State')

            # Add count labels
            for i, v in enumerate(state_counts['Count']):
                ax.text(v + 0.1, i, str(v), va='center')

            st.pyplot(fig)

            # Show data
            st.dataframe(data['flipped_dem_to_rep'])
        else:
            st.warning("Flipped Democrat to Republican data not available.")

    with col2:
        st.markdown("### Counties that Flipped Republican to Democrat")
        if data['flipped_rep_to_dem'] is not None:
            # Group by state and count
            state_counts = data['flipped_rep_to_dem']['state_po'].value_counts().reset_index()
            state_counts.columns = ['State', 'Count']

            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, len(state_counts) * 0.4 + 2))
            ax.barh(state_counts['State'], state_counts['Count'], color='blue')
            ax.set_xlabel('Number of Counties')
            ax.set_title('Counties Flipped Rep → Dem by State')

            # Add count labels
            for i, v in enumerate(state_counts['Count']):
                ax.text(v + 0.1, i, str(v), va='center')

            st.pyplot(fig)

            # Show data table
            st.dataframe(data['flipped_rep_to_dem'])
        else:
            st.warning("Flipped Republican to Democrat data not available.")

    # Margin Change Analysis
    st.markdown("### Margin Change Analysis")

    if data['flipped_rep_to_dem'] is not None and data['flipped_dem_to_rep'] is not None:
        col1, col2 = st.columns(2)

        with col1:
            avg_margin_rep_to_dem = data['flipped_rep_to_dem']['margin_change'].mean()
            st.metric("Average Margin Change (Rep → Dem)", f"{avg_margin_rep_to_dem:.2f}%")

            # Distribution of margin changes
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data['flipped_rep_to_dem']['margin_change'], bins=15, color='blue', alpha=0.7)
            ax.axvline(avg_margin_rep_to_dem, color='black', linestyle='--',
                       label=f'Mean: {avg_margin_rep_to_dem:.2f}%')
            ax.set_xlabel('Margin Change (%)')
            ax.set_ylabel('Number of Counties')
            ax.set_title('Distribution of Margin Changes (Rep → Dem)')
            ax.legend()

            st.pyplot(fig)

        with col2:
            avg_margin_dem_to_rep = data['flipped_dem_to_rep']['margin_change'].mean()
            st.metric("Average Margin Change (Dem → Rep)", f"{avg_margin_dem_to_rep:.2f}%")

            # Distribution of margin changes
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data['flipped_dem_to_rep']['margin_change'], bins=10, color='red', alpha=0.7)
            ax.axvline(avg_margin_dem_to_rep, color='black', linestyle='--',
                       label=f'Mean: {avg_margin_dem_to_rep:.2f}%')
            ax.set_xlabel('Margin Change (%)')
            ax.set_ylabel('Number of Counties')
            ax.set_title('Distribution of Margin Changes (Dem → Rep)')
            ax.legend()

            st.pyplot(fig)
    else:
        st.warning("Flipped counties data not available.")

# About Page
elif page == "About":
    st.markdown('<div class="sub-header">About This Dashboard</div>', unsafe_allow_html=True)

    st.markdown("""
    ## 🎓 Ballot Box to Data Point: Tracking America's Political Pulse

    In the wake of recent political upheaval, this dashboard visualizes voting pattern shifts across U.S. counties from 2016 to 2020. 
    This analytical tool transforms raw electoral data into actionable insights, highlighting how 57 counties flipped from Republican 
    to Democratic while 11 shifted from Democratic to Republican.

    The dashboard showcases dramatic margin shifts like Inyo County, California's 13.7% Democratic gain and Kenedy County, 
    Texas's stunning 40% Republican surge. Drawing from county-level electoral databases and demographic indicators, 
    this visualization suite helps campaign strategists, political scientists, and engaged citizens understand 
    America's evolving political landscape.

    ### Data Sources:
    - **MIT Election Data and Science Lab** (Presidential election results at the county level, 2000–2020)
    - **U.S. Census Bureau** (County-level education attainment percentages across years)

    ### Project Goals:
    - Analyze how education levels correlate with political leanings at the state level
    - Investigate trends over time across multiple election cycles
    - Provide visualizations and correlations between higher education rates and Democratic margins
    """)

    # Show loaded datasets
    st.markdown("### Available Datasets")

    loaded_data = {key: df is not None for key, df in data.items()}
    loaded_df = pd.DataFrame({
        'Dataset': loaded_data.keys(),
        'Loaded': loaded_data.values()
    })

    st.dataframe(loaded_df)

    # Project structure
    st.markdown("### Project Structure")
    st.code("""
voter-turnout-predictor/
├── src/
│   ├── analysis/
│   │   ├── education_analysis.py
│   │   ├── educational_deep_dive.py
│   │   ├── state_pivot.py
│   │   └── ...
│   ├── data/
│   │   ├── analysis/
│   │   │   ├── flipped_dem_to_rep_2016_2020.csv
│   │   │   ├── flipped_rep_to_dem_2016_2020.csv
│   │   │   └── ...
│   │   ├── processed/
│   │   │   ├── state_education_summary.csv
│   │   │   ├── state_pivot_summary.csv
│   │   │   └── ...
│   │   ├── raw/
│   │   │   ├── census_county_data_all_years.csv
│   │   │   ├── mit_election_lab_county_returns_raw.csv
│   │   │   └── ...
│   ├── data_collection/
│   │   ├── census_data_collector.py
│   │   ├── election_data_collector.py
│   │   └── ...
│   └── reporting/
│       ├── visualization.py
│       └── ...
├── app.py
├── requirements.txt
└── README.md
""")

# Add footer
st.markdown("---")
st.markdown("© 2025 | Education & Voting Patterns Analyzer | Version 1.0")