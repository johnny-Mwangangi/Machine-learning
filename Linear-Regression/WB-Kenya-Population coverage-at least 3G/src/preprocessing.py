# src/preprocessing.py

import pandas as pd

def merge_features(
        coverage: pd.DataFrame,
        mobile: pd.DataFrame,
        population: pd.DataFrame,
):
    df=(
        coverage
        .merge(mobile, on="year", how="left")
        .merge(population, on="year", how="left")
        .dropna()
    )

    return df
