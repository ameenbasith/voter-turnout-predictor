import os
import pandas as pd


def find_files_in_project():
    """
    Search for data files in the project directory and subdirectories.
    Returns a dictionary of found files and their paths.
    """
    project_root = os.getcwd()
    print(f"Current working directory: {project_root}")

    # Files we're looking for
    target_files = [
        'census_county_data_all_years.csv',
        'countypres_20002020.csv',
        'mit_election_lab_county_returns_raw.csv',
        'state_pivot_summary.csv',
        'state_education_summary.csv',
        'voter_turnout_features.csv',
        'flipped_rep_to_dem_2016_2020.csv',
        'flipped_dem_to_rep_2016_2020.csv',
        'top_dem_shifts.csv',
        'top_rep_shifts.csv',
        'partisan_lean_by_education_quintile.csv',
        'national_turnout_by_year.csv'
    ]

    found_files = {}

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file in target_files:
                rel_path = os.path.relpath(os.path.join(root, file), project_root)
                found_files[file] = rel_path
                print(f"Found {file} at {rel_path}")

    # Check for CSV files that might match our needs
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.csv') and file not in target_files:
                rel_path = os.path.relpath(os.path.join(root, file), project_root)
                if any(keyword in file.lower() for keyword in
                       ['county', 'election', 'state', 'education', 'vote', 'turn']):
                    print(f"Potential relevant CSV file: {file} at {rel_path}")

    return found_files


def preview_csv_files(found_files):
    """
    Preview the structure of found CSV files to understand their content.
    """
    print("\n=== CSV File Structure Preview ===")

    for filename, path in found_files.items():
        full_path = os.path.join(os.getcwd(), path)
        try:
            df = pd.read_csv(full_path, nrows=5)
            print(f"\nFile: {filename}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("Preview:")
            print(df.head(2))
        except Exception as e:
            print(f"Error reading {filename}: {e}")


if __name__ == "__main__":
    print("Searching for data files in your project...")
    found_files = find_files_in_project()

    print("\n=== Summary of Found Files ===")
    if found_files:
        for file, path in found_files.items():
            print(f"- {file}: {path}")

        # Preview CSV files
        preview_csv_files(found_files)
    else:
        print("No target data files found in the project directory.")

    print("\nTo run the Streamlit app, follow these steps:")
    print("1. Create a new file called 'simple_app.py' in your project root")
    print("2. Copy the Streamlit app code that will be generated for you")
    print("3. Run 'streamlit run simple_app.py'")