from pyexpat import features

from joblib import load
import pandas as pd

bundle = load("Models/kenya_3g_pipeline.joblib")
pipeline = bundle["pipeline"]
features = bundle["features"]

new_data = pd.DataFrame (
    {
        "mobile_subs_per_100": [85.0],
        "population_total": [60_000_000],
    }
)

prediction = pipeline.predict(new_data[features])
print("Predict 3G coverage:", prediction[0])

