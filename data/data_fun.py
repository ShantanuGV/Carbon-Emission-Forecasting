import pandas as pd
import numpy as np

def clean_colum():
    files = {
    "row_data/world8024.csv": "clean_data/co2.csv",
    "row_data/worldeco.csv": "clean_data/industry.csv",
    "row_data/worldenergy.csv": "clean_data/energy.csv",
    "row_data/worldforest.csv": "clean_data/forest.csv",
    "row_data/worldfos.csv": "clean_data/fossil.csv",
    "row_data/worldpop.csv": "clean_data/population.csv",
    "row_data/worldrenew.csv": "clean_data/renewable.csv",
    }

    for inp, out in files.items():
        clean_wdi_file(inp, out)

def clean_wdi_file(input_path, output_path):

    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        on_bad_lines="skip"
    )

    # Remove unnamed junk columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Drop useless metadata columns if present
    drop_cols = ["Indicator Name", "Indicator Code"]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    # Save cleaned file
    df.to_csv(output_path, index=False)

    print(f"Cleaned file saved → {output_path}")
    
def world_r(input_path):
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # Separate numeric columns
    sum_cols = ["Population"]
    avg_cols = ["Energy_PC"]

    # Create WORLD dataframe
    world_rows = []

    for year in df["Year"].unique():
        year_df = df[df["Year"] == year]

        world_row = {
            "Country Name": "World",
            "Country Code": "WLD",
            "Year": year,
        }

        # Sum columns
        for col in sum_cols:
            world_row[col] = year_df[col].sum(skipna=True)

        # Average columns
        for col in avg_cols:
            world_row[col] = year_df[col].mean(skipna=True)

        # CO₂ left blank

        world_rows.append(world_row)

    world_df = pd.DataFrame(world_rows)

    # Combine
    final_df = pd.concat([world_df, df], ignore_index=True)

    

    # Save new file (old safe)
    final_df.to_csv(input_path, index=False)

    print("WORLD row created ✅\n",final_df)    
    
    


import pandas as pd

def add_world_row(
    input_path,
    output_path=None,
    country_col="Country Name",
    code_col="Country Code",
    year_col="Year",
    population_col="Population",
    per_capita_cols=None
):
    """
    Correct WORLD calculation from per-capita data.
    Reconstructs totals first, then recomputes WORLD PC.
    """

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    if per_capita_cols is None:
        per_capita_cols = []

    world_rows = []

    for year in sorted(df[year_col].unique()):
        year_df = df[df[year_col] == year].copy()

        world_row = {
            country_col: "World",
            code_col: "WLD",
            year_col: year
        }

        # -------------------------
        # Population sum
        # -------------------------
        total_population = year_df[population_col].sum(skipna=True)
        world_row[population_col] = total_population

        # -------------------------
        # Per-capita reconstruction
        # -------------------------
        for col in per_capita_cols:

            valid = year_df[[population_col, col]].dropna()

            if len(valid) == 0:
                world_row[col] = None
                continue

            # Reconstruct total energy
            total_energy = (valid[col] * valid[population_col]).sum()

            # Compute world PC
            world_pc = total_energy / total_population

            world_row[col] = world_pc

        world_rows.append(world_row)

    world_df = pd.DataFrame(world_rows)

    final_df = pd.concat([world_df, df], ignore_index=True)

    if output_path is None:
        output_path = input_path

    final_df.to_csv(output_path, index=False)

    print(final_df.head())
    print("\n✅ WORLD rows added correctly")

    return final_df


    
    
    
def add_base_to_all(input_path,output_path):
    
    # Read population dataset
    df = pd.read_csv(input_path, encoding="utf-8-sig", on_bad_lines="skip")
    
    # Keep only identity columns
    base = df[["Country Name", "Country Code"]].copy()
    
    # Create year range
    years = list(range(1960, 2026))   # 1960 → 2025
    
    # Create cartesian product (Country × Year)
    base["key"] = 1
    year_df = pd.DataFrame({"Year": years})
    year_df["key"] = 1
    
    master_base = pd.merge(base, year_df, on="key").drop("key", axis=1)
    # Save file
    master_base.to_csv(output_path, index=False)
    
    print("Base dataframe created ✅")
    print(master_base)
        
        
def add_to_all(input_path, output_path, metric_name):
    
    # Read files
    df = pd.read_csv(input_path, encoding="utf-8-sig", on_bad_lines="skip")
    out_df = pd.read_csv(output_path, encoding="utf-8-sig")
    
    # Keep required columns
    df = df[["Country Name", "Country Code"] +
            [col for col in df.columns if col.isdigit()]]
    
    # Convert wide → long
    long_df = df.melt(
        id_vars=["Country Name", "Country Code"],
        var_name="Year",
        value_name=metric_name
    )
    
    # Convert year to int
    long_df["Year"] = long_df["Year"].astype(int)
    
    if metric_name in out_df.columns:
        out_df = out_df.drop(columns=[metric_name])


    
    # Merge with base
    merged = pd.merge(
        out_df,
        long_df,
        on=["Country Name", "Country Code", "Year"],
        how="left"
    )
    
    # Save
    merged.to_csv(output_path, index=False)
    
    print(f"{metric_name} added ✅")
    print(merged)            
    
    
    
def fill_population(group):

    group = group.copy()
    
    # Get non-null values
    not_null = group.dropna(subset=["Population"])
    
    if len(not_null) < 2:
        return group
    
    first_year = not_null.iloc[0]["Year"]
    last_year  = not_null.iloc[-1]["Year"]
    
    first_val = not_null.iloc[0]["Population"]
    last_val  = not_null.iloc[-1]["Population"]
    
    years_diff = last_year - first_year
    
    growth = (last_val - first_val) / years_diff if years_diff != 0 else 0
    
    
    # Fill nulls
    for i, row in group.iterrows():
        
        if pd.isna(row["Population"]):
            
            year = row["Year"]
            
            if year > last_year:
                diff = year - last_year
                group.at[i, "Population"] = last_val + growth * diff
                
            elif year < first_year:
                diff = first_year - year
                group.at[i, "Population"] = first_val - growth * diff
    
    return group


def metrics_pc(input_path,metrics):
    # Load file
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df[['Country Name','Country Code','Year','Population']+metrics]

    # Avoid divide by zero
    df["Population"] = df["Population"].replace(0, pd.NA)

    # Create per-capita columns
    for col in metrics:
        new_col = f"{col}_PC"
        df[new_col] = df[col] / df["Population"]

    df = df.drop(columns=metrics)

    # Save new file (safe)
    output_path = "all_data_with_pc.csv"
    df.to_csv(output_path, index=False)
    


    print("Per-capita columns created ✅")
    print("Saved as:", output_path)

    return df



def fill_early_history_energy_pc(input_path):

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["Year"] = df["Year"].astype(int)

    countries = df["Country Code"].unique()
    filled_count = 0

    for country in countries:

        c_df = df[df["Country Code"] == country] \
                .sort_values("Year") \
                .reset_index()

        pc_series = c_df["Energy_PC"]

        # First known value
        first_valid = pc_series.first_valid_index()
        if first_valid is None:
            continue

        # Next known value (to calculate growth)
        next_valid = pc_series[first_valid+1:].first_valid_index()
        if next_valid is None:
            continue

        v1 = pc_series[first_valid]
        v2 = pc_series[next_valid]

        y1 = c_df.loc[first_valid, "Year"]
        y2 = c_df.loc[next_valid, "Year"]

        year_diff = y2 - y1
        if v1 == 0 or year_diff == 0:
            continue

        growth = (v2 / v1) ** (1 / year_diff) - 1

        # Start reverse propagation
        last_val = v1

        for i in range(first_valid - 1, -1, -1):

            new_val = last_val / (1 + growth)

            df.loc[c_df.loc[i, "index"], "Energy_PC"] = new_val

            last_val = new_val
            filled_count += 1

    df.to_csv("all_data_with_pc.csv", index=False)
    

    print(f"Early history filled fully ✅ Rows filled: {filled_count}")
    
    
    
    
def interpolate_energy_pc(input_path):

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    countries = df["Country Code"].unique()
    filled_total = 0

    for country in countries:

        mask = df["Country Code"] == country

        before_nulls = df.loc[mask, "Energy_PC"].isna().sum()

        # Interpolate linearly across years
        df.loc[mask, "Energy_PC"] = (
            df.loc[mask]
            .sort_values("Year")["Energy_PC"]
            .interpolate(method="linear")
        )

        after_nulls = df.loc[mask, "Energy_PC"].isna().sum()

        filled_total += (before_nulls - after_nulls)

    df.to_csv("all_data_with_pc.csv", index=False)
    print(df)

    print(f"Random gaps filled ✅ Rows filled: {filled_total}")    
    
    
    
def fill_energy_pc_world_avg(input_path):

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    filled_total = 0

    # Calculate world average per year
    world_avg = (
        df.groupby("Year")["Energy_PC"]
        .mean()
        .reset_index()
        .rename(columns={"Energy_PC": "World_Energy_PC"})
    )

    # Merge world avg back
    df = df.merge(world_avg, on="Year", how="left")

    # Fill remaining nulls
    mask = df["Energy_PC"].isna()

    filled_total = mask.sum()

    df.loc[mask, "Energy_PC"] = df.loc[mask, "World_Energy_PC"]

    # Drop helper column
    df.drop(columns=["World_Energy_PC"], inplace=True)

    df.to_csv("all_data_with_pc.csv", index=False)
    
    print(df)

    print(f"No-data countries filled 🌍 Rows filled: {filled_total}")


def fill_no_energy_pc_countries(all_data_path, save=True):

    df = pd.read_csv(all_data_path, encoding="utf-8-sig")

    # -----------------------------------
    # 1. Countries with no PC data
    # -----------------------------------
    country_pc_sum = (
        df.groupby("Country Name")["Energy_PC"]
        .sum(min_count=1)
    )

    no_pc_countries = country_pc_sum[
        country_pc_sum.isna() | (country_pc_sum == 0)
    ].index.tolist()

    print("No-data countries:", len(no_pc_countries))

    # -----------------------------------
    # 2. Compute SAFE World PC
    # -----------------------------------
    world_pc_list = []

    for year in sorted(df["Year"].unique()):

        year_df = df[
            (df["Year"] == year) &
            (df["Country Name"] != "World") &
            (df["Energy_PC"] > 0)
        ]

        # Require minimum countries
        if len(year_df) >= 20:
            pc_value = year_df["Energy_PC"].median()
        else:
            pc_value = np.nan

        world_pc_list.append((year, pc_value))

    world_pc = pd.DataFrame(
        world_pc_list,
        columns=["Year", "World_Energy_PC"]
    )

    # Interpolate missing early years
    world_pc["World_Energy_PC"] = (
        world_pc["World_Energy_PC"]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    # Merge
    df = df.merge(world_pc, on="Year", how="left")

    # -----------------------------------
    # 3. Fill missing countries
    # -----------------------------------
    fill_rows = 0

    for country in no_pc_countries:

        mask = df["Country Name"] == country

        df.loc[mask, "Energy_PC"] = df.loc[
            mask, "World_Energy_PC"
        ]

        fill_rows += mask.sum()

    print("Rows filled:", fill_rows)

    # Cleanup
    df.drop(columns=["World_Energy_PC"], inplace=True)

    if save:
        df.to_csv(all_data_path, index=False)
        print("Dataset updated ✅")

    return df