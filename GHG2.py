import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

excel_file = '"C:\\Users\\Kaustab das\\Desktop\\GHG\\SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx"'  # Replace with actual path
years = range(2010, 2017)
years[0]
df_1 = pd.read_excel(excel_file, sheet_name=f'{years[0]}_Detail_Commodity')
df_1.head()
df_2 = pd.read_excel(excel_file, sheet_name=f'{years[0]}_Detail_Industry')
df_2.head()
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

        df_com.rename(columns={
            'Commodity Code': 'Code',
            'Commodity Name': 'Name'
        }, inplace=True)
        
        df_ind.rename(columns={
            'Industry Code': 'Code',
            'Industry Name': 'Name'
        }, inplace=True)
        
        all_data.append(pd.concat([df_com, df_ind], ignore_index=True))
        
    except Exception as e:
        print(f"Error processing year {year}: {e}")

all_data[3]
len(all_data)
df = pd.concat(all_data, ignore_index=True)
df.head()
len(df)
df.columns # Checking columns

df.isnull().sum()
# As there is no data avaialble in Unnamed coulmn so we will drop the column
df.drop(columns=['Unnamed: 7'],inplace=True)

print(df.info())   # Checking data types and non-null counts

df.describe().T # Checking summary statistics

df.isnull().sum() # Checking for null values in each column 

# Visualize distribution
sns.histplot(df['Supply Chain Emission Factors with Margins'], bins=50, kde=True)
plt.title('Target Variable Distribution')
plt.show()
# Check categorical variables
print(df['Substance'].value_counts())
print(df['Unit'].value_counts()) # Checking unique values in 'Unit' with count 
print(df['Unit'].unique()) # Checking unique values in 'Unit'

print(df['Source'].value_counts()) # Checking unique values in 'Source' with count 

df['Substance'].unique() # Checking unique values in 'Substance' 
substance_map={'carbon dioxide':0, 'methane':1, 'nitrous oxide':2, 'other GHGs':3} # Mapping substances to integers
df['Substance']=df['Substance'].map(substance_map)  
df['Substance'].unique() # Checking unique values in 'Substance' 

print(df['Unit'].unique()) # Checking unique values in 'Unit'
unit_map={'kg/2018 USD, purchaser price':0, 'kg CO2e/2018 USD, purchaser price':1} # Mapping units to integers

df['Unit']=df['Unit'].map(unit_map) 
print(df['Unit'].unique()) # Checking unique values in 'Unit' 
print(df['Source'].unique()) # Checking unique values in 'Source' 
source_map={'Commodity':0, 'Industry':1} # Mapping sources to integers 
df['Source']=df['Source'].map(source_map)   # applying the mapping to 'Source' column 

print(df['Source'].unique()) # Checking unique values in 'Source' 

df.info() # Checking data types and non-null counts after mapping 

df.Code.unique() # Checking unique values in 'Code' df['Code']
df.Name.unique() # Checking unique values in 'Name' 

len(df.Name.unique()) # Checking number of unique values in 'Name'

top_emitters = df[['Name', 'Supply Chain Emission Factors with Margins']].groupby('Name').mean().sort_values(
    'Supply Chain Emission Factors with Margins', ascending=False).head(10) 

# Resetting index for better plotting
top_emitters = top_emitters.reset_index()

top_emitters
# Plotting the top 10 emitting industries


plt.figure(figsize=(10,6))
# Example: Top emitting industries (already grouped)
sns.barplot(
    x='Supply Chain Emission Factors with Margins',
    y='Name',
    data=top_emitters,
    hue='Name',
    palette='viridis'  # Use 'Blues', 'viridis', etc., for other color maps
)

# Add ranking labels (1, 2, 3...) next to bars
for i, (value, name) in enumerate(zip(top_emitters['Supply Chain Emission Factors with Margins'], top_emitters.index), start=1):
    plt.text(value + 0.01, i - 1, f'#{i}', va='center', fontsize=11, fontweight='bold', color='black')

plt.title('Top 10 Emitting Industries', fontsize=14, fontweight='bold') # Title of the plot 
plt.xlabel('Emission Factor (kg CO2e/unit)') # X-axis label
plt.ylabel('Industry') # Y-axis label
plt.grid(axis='x', linestyle='--', alpha=0.6) # Adding grid lines for better readability
plt.tight_layout() # Adjust layout to prevent overlap

plt.show()
df.drop(columns=['Name','Code','Year'], inplace=True) 

df.shape

df.columns
X = df.drop(columns=['Supply Chain Emission Factors with Margins']) # Feature set excluding the target variable
y = df['Supply Chain Emission Factors with Margins'] # Target variable 
X.head()
y.head()

# Count plot for Substance
plt.figure(figsize=(6, 3))
sns.countplot(x=df["Substance"])
plt.title("Count Plot: Substance")
plt.xticks()
plt.tight_layout()
plt.show()
# Count plot for Unit
plt.figure(figsize=(6, 3))
sns.countplot(x=df["Unit"])
plt.title("Count Plot: Unit")
plt.tight_layout()
plt.show()
# Count plot for Source
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Source"])
plt.title("Count Plot: Source (Industry vs Commodity)")
plt.tight_layout()
plt.show()
df.columns
df.select_dtypes(include=np.number).corr() # Checking correlation between numerical features 

df.info() # Checking data types and non-null counts after mapping 
# Correlation matrix 
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()