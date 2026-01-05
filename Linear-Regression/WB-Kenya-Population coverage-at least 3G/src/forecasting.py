import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def forecast_feature(df, feature: str, future_years):
    model = LinearRegression()
    model.fit(df[["year"]], df[feature])

    future_df = pd.DataFrame({"year": list(future_years)})
    future_df[feature] = model.predict(future_df[["year"]])

    return future_df

