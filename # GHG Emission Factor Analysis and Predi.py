# GHG Emission Factor Analysis and Prediction

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === 1. Load & Combine Excel Sheets from Multiple Years === #

excel_file = "C:\\Users\\Kaustab das\\Desktop\\GHG\\SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx"
years = range(2010, 2017)
all_data = []

for year in years:
    try:
        df_com = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Commodity')
        df_ind = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Industry')

        df_com['Source'] = 'Commodity'
        df_ind['Source'] = 'Industry'
        df_com['Year'] = df_ind['Year'] = year

        df_com.columns = df_com.columns.str.strip()
        df_ind.columns = df_ind.columns.str.strip()

        df_com.rename(columns={'Commodity Code': 'Code', 'Commodity Name': 'Name'}, inplace=True)
        df_ind.rename(columns={'Industry Code': 'Code', 'Industry Name': 'Name'}, inplace=True)

        combined = pd.concat([df_com, df_ind], ignore_index=True)
        all_data.append(combined)

    except Exception as e:
        print(f"Error processing year {year}: {e}")

# Concatenate all years' data
df = pd.concat(all_data, ignore_index=True)

# === 2. Data Cleaning & Feature Engineering === #

# Drop unnecessary column if exists
if 'Unnamed: 7' in df.columns:
    df.drop(columns=['Unnamed: 7'], inplace=True)

# Map categorical variables to numeric
substance_map = {'carbon dioxide': 0, 'methane': 1, 'nitrous oxide': 2, 'other GHGs': 3}
unit_map = {'kg/2018 USD, purchaser price': 0, 'kg CO2e/2018 USD, purchaser price': 1}
source_map = {'Commodity': 0, 'Industry': 1}

df['Substance'] = df['Substance'].map(substance_map)
df['Unit'] = df['Unit'].map(unit_map)
df['Source'] = df['Source'].map(source_map)

# Drop high-cardinality or unused columns
df.drop(columns=['Name', 'Code', 'Year'], inplace=True)

# Drop any rows with missing data (optional)
df.dropna(inplace=True)

# === 3. EDA (Visualization) === #

# Distribution of target
plt.figure(figsize=(8, 4))
sns.histplot(df['Supply Chain Emission Factors with Margins'], bins=50, kde=True)
plt.title('Distribution of Emission Factors')
plt.xlabel('Emission Factor (kg CO2e/unit)')
plt.tight_layout()
plt.show()

# Count plots for categorical variables
for col in ['Substance', 'Unit', 'Source']:
    plt.figure(figsize=(5, 3))
    sns.countplot(x=col, data=df)
    plt.title(f'Count Plot: {col}')
    plt.tight_layout()
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# === 4. Prepare Features and Target === #

X = df.drop(columns=['Supply Chain Emission Factors with Margins'])
y = df['Supply Chain Emission Factors with Margins']

# === 5. Train-Test Split & Scaling === #

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. Train Random Forest Model === #

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === 7. Evaluate Model === #

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# === 8. Save Model and Scaler === #

output_dir = "C:\\Users\\Kaustab das\\Desktop\\GHG\\model"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model, os.path.join(output_dir, 'emission_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

print("✅ Model and scaler saved successfully.")
