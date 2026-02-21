import pandas as pd
import joblib

# -----------------------------------------
# 1. Load test feature data
# -----------------------------------------

test_df = pd.read_csv("test_features.csv")

print("Test data shape:", test_df.shape)

# -----------------------------------------
# 2. Separate features and target
# -----------------------------------------

target = "runs"

X_test = test_df.drop(columns=[target])

# Keep only numeric columns (same as training)
X_test = X_test.select_dtypes(include=["int64", "float64"])

# -----------------------------------------
# 3. Load trained LightGBM model
# -----------------------------------------

model = joblib.load("lgb_model.pkl")

print("Model loaded successfully")

# -----------------------------------------
# 4. Predict runs
# -----------------------------------------

predicted_runs = model.predict(X_test)

# -----------------------------------------
# 5. Create result dataframe
# -----------------------------------------

results = test_df.copy()
results["predicted_runs"] = predicted_runs

# -----------------------------------------
# 6. Save final predictions
# -----------------------------------------

results.to_csv("week7_predictions.csv", index=False)

print("Predictions saved to week7_predictions.csv")

# -----------------------------------------
# 7. Show few predictions
# -----------------------------------------

print(results[["runs", "predicted_runs"]].head(10))
