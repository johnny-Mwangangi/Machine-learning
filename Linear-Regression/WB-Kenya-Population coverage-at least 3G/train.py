from joblib import dump
from src.config import FEATURE_COLS, TARGET_COL, FUTURE_YEARS
from src.data_loader import load_3g_coverage, load_wb_dataset
from src.preprocessing import merge_features
from src.modeling import train_and_evaluate
from src.forecasting import forecast_feature
from src.visualization import (
plot_forecast,
plot_coverage_trend,
plot_predictor_vs_target,
plot_actual_vs_predicted
)

def main():
    coverage = load_3g_coverage("WB-Dataset/ITU_DH_POP_COV_3G_WIDEF.csv")
    mobile = load_wb_dataset(
        "WB-Dataset/API_IT.CEL.SETS.P2_DS2_en_csv_v2_3030.csv",
        "mobile_subs_per_100",
    )
    population = load_wb_dataset(
        "WB-Dataset/API_SP.POP.TOTL_DS2_en_csv_v2_69.csv",
        "population_total",
    )

    df = merge_features(coverage, mobile, population)

    plot_coverage_trend(df)
    plot_predictor_vs_target(df, "mobile_subs_per_100")
    plot_predictor_vs_target(df, "population_total")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    pipeline, metrics = train_and_evaluate(X, y)
    print("Model Performance:", metrics)

    #Actualvspredicted
    y_pred = pipeline.predict(X)
    plot_actual_vs_predicted(y, y_pred)

    #Forecast predictors
    future_mobile = forecast_feature(df, "mobile_subs_per_100", FUTURE_YEARS)
    future_population = forecast_feature(df, "population_total", FUTURE_YEARS)

    future_df = (
        future_mobile
        .merge(future_population, on="year")
    )

    #Predict future 3G Coverage
    future_df["predicted_3G"] = pipeline.predict(
        future_df[FEATURE_COLS]
    )

    #forecast plots
    plot_forecast(df, future_df)

    dump(
        {
            "pipeline": pipeline,
            "features": FEATURE_COLS,
        },
        "Models/kenya_3g_pipeline.joblib",
    )

if __name__ == "__main__":
    main()

