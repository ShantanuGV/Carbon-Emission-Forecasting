import pandas as pd
import glob
import json

files = glob.glob('data/row_data/*.csv')
res = {}
for f in files:
    try:
        df = pd.read_csv(f, on_bad_lines='skip', skiprows=4)
        if 'Indicator Name' in df.columns:
            res[f.split('\\')[-1]] = df['Indicator Name'].iloc[0]
        else:
            df2 = pd.read_csv(f, on_bad_lines='skip')
            res[f.split('\\')[-1]] = df2['Indicator Name'].iloc[0] if 'Indicator Name' in df2.columns else 'No Indicator Name'
    except Exception as e:
        res[f.split('\\')[-1]] = str(e)

with open('indicators.json', 'w') as f:
    json.dump(res, f, indent=4)
