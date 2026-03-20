import pandas as pd
import numpy as np

# Mappings for the World Bank CSVs still needed for factors
mappings = {
    "worldeco.csv": "Industrial_Growth",
    "worldenergy.csv": "Energy_Demand_Index",
    "worldforest.csv": "Forest_Cover_Percent",
    "worldfos.csv": "Fossil_Percent",
    "worldpop.csv": "Population_Million",
    "worldrenew.csv": "Renewable_Percent"
}

dfs = []

# --- 1. Get CO2 from OWID ---
co2_path = "data/row_data/owid_co2.csv"
df_co2_raw = pd.read_csv(co2_path)
# Use 'co2_including_luc' to match higher Google/Global Carbon Project figures (~41.6bn)
target_col = 'co2_including_luc' if 'co2_including_luc' in df_co2_raw.columns else 'co2'
df_co2_world = df_co2_raw[df_co2_raw['country'] == 'World'][['year', target_col]]
df_co2_world.columns = ['Year', 'Emission']
df_co2_world = df_co2_world.set_index('Year')
dfs.append(df_co2_world)

# --- 2. Get Factors from World Bank files ---
for file_path, col_name in mappings.items():
    path = f"data/row_data/{file_path}"
    try:
        df = pd.read_csv(path, on_bad_lines='skip', skiprows=4)
        if 'Indicator Name' not in df.columns:
            df = pd.read_csv(path, on_bad_lines='skip')
    except:
        df = pd.read_csv(path, on_bad_lines='skip')
    
    # Focus on World
    df_world = df[df['Country Name'] == 'World']
    if df_world.empty:
        df_world = df[df['Country Code'] == 'WLD']
        
    if df_world.empty:
        years_cols = [c for c in df.columns if c.isdigit()]
        if col_name in ['Population_Million']:
            df_series = df[years_cols].sum(skipna=True)
        else:
            df_series = df[years_cols].mean(skipna=True)
    else:
        years_cols = [c for c in df_world.columns if c.isdigit()]
        df_series = df_world[years_cols].iloc[0]
        
    df_series.name = col_name
    
    # Unit conversion
    if col_name == 'Population_Million':
        df_series = df_series / 1000000.0
        
    df_factor = df_series.to_frame()
    df_factor.index.name = 'Year'
    df_factor.index = df_factor.index.astype(int)
    dfs.append(df_factor)

# --- 3. Join everything ---
final_df = pd.concat(dfs, axis=1)
final_df = final_df.reset_index()
final_df['Year'] = final_df['Year'].astype(int)

# Filter for relevant years
final_df = final_df[final_df['Year'] >= 1990]

# --- 4. Interpolate missing ---
final_df = final_df.sort_values('Year')
cols_to_fill = list(mappings.values()) + ['Emission']
final_df[cols_to_fill] = final_df[cols_to_fill].interpolate(method='linear', limit_direction='both')

# --- 5. Custom derived columns ---
final_df['Industrial_Production_Index'] = 100.0
curr = 100.0
for idx, row in final_df.iterrows():
    if not np.isnan(row['Industrial_Growth']):
        curr = curr * (1 + row['Industrial_Growth'] / 100.0)
    final_df.at[idx, 'Industrial_Production_Index'] = curr

# Scale Energy_Demand_Index to be 100 around 1990
if 1990 in final_df['Year'].values:
    base_val = final_df[final_df['Year'] == 1990]['Energy_Demand_Index'].values[0]
    if np.isnan(base_val) or base_val == 0:
        base_val = final_df['Energy_Demand_Index'].mean()
else:
    base_val = final_df['Energy_Demand_Index'].mean()

final_df['Energy_Demand_Index'] = (final_df['Energy_Demand_Index'] / base_val) * 100.0
final_df['Transport_Index'] = final_df['Energy_Demand_Index'] * 1.05
final_df['Urbanization_Rate'] = 33.0 + (final_df['Year'] - 1960) * (56.0 - 33.0) / (2020 - 1960)

# Final Cleanup
final_df = final_df.fillna(method='bfill').fillna(method='ffill')
final_df = final_df.dropna(subset=['Emission']) # Must have emission

output_path = 'data/real_emission_dataset.csv'
final_df.to_csv(output_path, index=False)
print("Data created successfully at", output_path)
print(final_df.tail())
if not final_df.empty:
    print("Latest Emission (2022+):", final_df.iloc[-1]['Emission'])
