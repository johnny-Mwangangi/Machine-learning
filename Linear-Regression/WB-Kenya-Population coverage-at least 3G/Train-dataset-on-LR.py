from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Drop rows with missing data
df_clean = df_model.dropna()

X = df_clean[['mobile_subs_per_100','population_total']]
y = df_clean['coverage_3G']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R² score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Show coefficients
coeffs = pd.Series(model.coef_, index=X.columns)
print("\nModel coefficients:")
print(coeffs)
