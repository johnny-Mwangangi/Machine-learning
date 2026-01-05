#src/visualization.py

import matplotlib.pyplot as plt
from scipy.ndimage import label

def plot_coverage_trend(df):
    plt.figure(figsize=(10,4))
    plt.plot(df["year"], df["coverage_3g"], marker="o")
    plt.title("Kenya - 3G Coverage Over Time")
    plt.xlabel("Year")
    plt.ylabel("3G Coverage (%)")
    plt.grid(True)
    plt.show()

def plot_predictor_vs_target(df, predictor):
    plt.figure(figsize=(6,4))
    plt.scatter(df[predictor], df["coverage_3g"])
    plt.xlabel(predictor.replace("_"," ").title())
    plt.ylabel("3G Coverage (%)")
    plt.title(f"{predictor.replace('_',' ').title()} vs 3G Coverage")
    plt.grid(True)
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(6,4))
    plt.scatter(y_true, y_pred)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
    )
    plt.xlabel("Actual 3G Coverage (%)")
    plt.ylabel("Predicted 3G Coverage (%)")
    plt.title("Actual vs Predicted 3G Coverage")
    plt.grid(True)
    plt.show()

def plot_forecast(df, future_df, show:bool=True):
    plt.figure(figsize=(10,5))
    plt.plot(df['year'], df['coverage_3g'], marker='o', label='Actual')
    plt.plot(future_df['year'], future_df['predicted_3G'],marker='o', label='Forecast (2025–2030)')
    plt.title("Kenya 3G Coverage — Actual & Forecast (2025–2030)")
    plt.xlabel("Year")
    plt.ylabel("3G Coverage (%)")
    plt.grid(True)
    plt.legend()
    plt.show()