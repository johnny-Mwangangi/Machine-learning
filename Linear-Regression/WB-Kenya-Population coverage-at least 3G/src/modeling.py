# src/modeling.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

def build_pipeline():
    return Pipeline(
        steps=[
            ("model", LinearRegression())
        ]
    )

def train_and_evaluate(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.25, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred)
    }

    return pipeline, metrics