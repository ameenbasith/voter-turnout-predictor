
---

## 🧠 Project Objective

- Analyze **how education levels** (Bachelor's, Master's, Doctorate) correlate with **political leanings** at the **state level**.
- Investigate **trends over time** (2012, 2016, 2020 elections).
- Provide **visualizations** and **correlations** between higher education rates and Democratic margins.

---

## 📊 Data Sources

- **MIT Election Data and Science Lab**  
  (Presidential election results at the county level, 2000–2020)
- **U.S. Census Bureau**  
  (County-level education attainment percentages across years)

---

## 🔥 Main Features

- **Education Analysis**:  
  Calculate higher education percentages across states and years.

- **Election Aggregation**:  
  Aggregate Democrat vs. Republican vote shares at the **state level**.

- **State Pivot Analysis**:  
  Merge education data with election results and correlate education with Democratic margin.

- **Visualizations**:  
  Generate scatter plots showing relationships between education levels and political leaning.

- **Reporting**:  
  Automatically save findings and charts into organized folders.

---

## 🚀 How to Run the Project

1. **Install required packages**  
   (create a `.venv` environment and install dependencies from `requirements.txt`)

2. **Organize the data**  
   - Place raw data into `data/raw/`
   - Clean datasets will be saved into `data/processed/`

3. **Run the Main Analysis**
   ```bash
   python src/main.py
