import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the processed data
DATA_PATH = "data/processed/merged_turnout_data.csv"
df = pd.read_csv(DATA_PATH)

# Define feature columns and target
features = [
    'total_population',
    'median_household_income',
    'higher_education_percentage',
    'is_swing_state',
    'total_ad_spend',
    'republican_ad_spend',
    'democrat_ad_spend'
]
target = 'turnout_percentage'

# 🪪 Show sample and value stats before filtering
print("\n📟 Sample values for features + target:")
print(df[features + [target]].head(10))

print("\n🪪 Column Value Summary BEFORE dropna:")
for col in features + [target]:
    print(f"{col}: non-zero count = {(df[col] != 0).sum()}, unique values = {df[col].nunique()}")

# Drop rows with missing values in features or target
df = df[features + [target]].dropna()

# 🔍 Show values after filtering
print("\n🔍 Sample after dropna:")
print(df.head(5))

# Ensure categorical/binary columns are correct
if 'is_swing_state' in df.columns:
    df['is_swing_state'] = df['is_swing_state'].astype(int)

# Prepare feature matrix and target vector
X = df[features]
y = df[target]

# Diagnostic output
print("\n✅ Dataset shape before training:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Missing values per column:\n", df.isnull().sum())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n✅ Model Evaluation")
print(f"R^2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} percentage points")

# Feature importance
importances = model.feature_importances_
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title("Feature Importance in Voter Turnout Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()

# Save figure
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/feature_importance.png")
plt.show()