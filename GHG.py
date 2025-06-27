# === Step 1: Library Imports ===
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === Step 2: File Setup ===
excel_path = r"C:\\Users\\Kaustab das\\Desktop\\GHG\\SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx"
year_range = list(range(2010, 2017))
dataset_merged = []

# === Step 3: Preview One Year's Sheets ===
first_year = year_range[0]
sheet_com = f'{first_year}_Detail_Commodity'
sheet_ind = f'{first_year}_Detail_Industry'

commodity_sample = pd.read_excel(excel_path, sheet_name=sheet_com, engine='openpyxl')
industry_sample = pd.read_excel(excel_path, sheet_name=sheet_ind, engine='openpyxl')

print(f"\n🎯 Preview - Commodity Sheet ({sheet_com}):\n", commodity_sample.head())
print(f"\n🎯 Preview - Industry Sheet ({sheet_ind}):\n", industry_sample.head())

# === Step 4: Combine Data from All Years ===
for yr in year_range:
    try:
        com_df = pd.read_excel(excel_path, sheet_name=f'{yr}_Detail_Commodity', engine='openpyxl')
        ind_df = pd.read_excel(excel_path, sheet_name=f'{yr}_Detail_Industry', engine='openpyxl')

        # Standardize source and year columns
        com_df['Source_Type'] = 'Commodity'
        ind_df['Source_Type'] = 'Industry'
        com_df['Year'] = ind_df['Year'] = yr

        # Trim whitespace from column names
        com_df.columns = com_df.columns.str.strip()
        ind_df.columns = ind_df.columns.str.strip()

        # Rename key identifiers
        com_df.rename(columns={
            'Commodity Code': 'Sector_Code',
            'Commodity Name': 'Sector_Name'
        }, inplace=True)

        ind_df.rename(columns={
            'Industry Code': 'Sector_Code',
            'Industry Name': 'Sector_Name'
        }, inplace=True)

        # Merge yearly data
        yearly_df = pd.concat([com_df, ind_df], axis=0, ignore_index=True)
        dataset_merged.append(yearly_df)

    except Exception as err:
        print(f"⚠️ Failed to process year {yr}: {err}")

# === Step 5: Final Combined Dataset ===
full_df = pd.concat(dataset_merged, ignore_index=True)
full_df.columns = full_df.columns.str.strip()  # Re-trim just in case

# === Step 6: Dataset Summary ===
print("\n✅ Combined Dataset Preview:")
print(full_df.head(10))

print(f"\n📊 Total Data Points: {len(full_df)}")

print("\n🧾 Available Columns:")
print(full_df.columns.tolist())

print("\n🧹 Missing Value Summary:")
print(full_df.isnull().sum())