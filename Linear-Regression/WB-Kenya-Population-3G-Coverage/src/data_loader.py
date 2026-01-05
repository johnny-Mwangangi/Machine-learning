# src/data_loader.py

import pandas as pd

def load_3g_coverage(path: str):
    df = pd.read_csv(path)
    df = df[df["REF_AREA_LABEL"]=="Kenya"]

    year_cols = [c for c in df.columns if c.startswith("20")]
    df = df[year_cols].transpose().reset_index()
    df.columns = ["variable", "coverage_3g"]

    df["year"] = df["variable"].str.extract(r"(\d{4})").astype(int)
    df = df[["year", "coverage_3g"]].dropna().sort_values("year")

    return df

def load_wb_dataset(path: str, value_name: str):
    df = pd.read_csv(path, skiprows=4)
    df = df[df['Country Name']=="Kenya"]

    year_cols = [c for c in df.columns if c.isdigit()]
    df = df.melt(
        id_vars=["Country Name"],
        value_name= value_name,
        var_name="year",
        value_vars=year_cols
    )

    df["year"] = df["year"].astype(int)
    return df[["year", value_name]].dropna()




