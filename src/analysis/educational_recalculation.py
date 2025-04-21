import pandas as pd
import os
import matplotlib.pyplot as plt

# === 1. Setup ===
education_path = '../data/processed/state_education_summary.csv'
election_path = '../data/processed/state_pivot_summary.csv'  # <-- wherever your state_pivot output is saved

# === 2. Load data ===
print("✅ Loading datasets...")
edu = pd.read_csv(education_path)
election = pd.read_csv(election_path)

print(f"🎓 Education data shape: {edu.shape}")
print(f"🗳️ Election data shape: {election.shape}")

# === 3. Create higher education total percentage ===
edu['higher_education_total_pct'] = (
    edu['bachelors_degree_pct'] +
    edu['masters_degree_pct'] +
    edu['professional_degree_pct'] +
    edu['doctorate_degree_pct']
)

print("✅ Created 'higher_education_total_pct' feature.")

# === 4. Merge education and election ===
edu['state'] = edu['state'].astype(str).str.zfill(2)  # make sure state codes are 2 digits
election['state_po'] = election['state_po'].astype(str).str.zfill(2)

merged = pd.merge(
    edu,
    election,
    how='left',
    left_on=['year', 'state'],
    right_on=['year', 'state_po']
)

print(f"✅ Merged dataset: {merged.shape}")

# === 5. Analyze correlation each year ===
for year in [2012, 2016, 2020]:
    df_year = merged[merged['year'] == year]

    if df_year.empty:
        print(f"⚠️ No data for year {year}")
        continue

    corr = df_year['higher_education_total_pct'].corr(df_year['dem_margin'])

    print(f"\n📅 Year {year}:")
    print(f"📈 Correlation between Total Higher Education % and Democratic Margin: {corr:.4f}")

    # === 6. Plotting ===
    plt.figure(figsize=(8,6))
    plt.scatter(
        df_year['higher_education_total_pct'],
        df_year['dem_margin'],
        alpha=0.7,
        edgecolor='k'
    )
    plt.title(f"Higher Education vs Democratic Margin ({year})", fontsize=16)
    plt.xlabel('Higher Education Total %', fontsize=14)
    plt.ylabel('Democratic Margin (%)', fontsize=14)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = f'reports/figures/education_vs_margin_{year}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    print(f"🖼️ Saved scatter plot to {plot_path}")

print("\n✅ Full analysis complete!")

