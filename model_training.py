import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load dataset
df = pd.read_csv("data/cleaned/dataset.csv")

X = df.drop(columns=['next_match_runs'])
y = df['next_match_runs']

# Load preprocessing pipeline
with open("models/feature_pipeline.pkl", "rb") as f:
    preprocessor = pickle.load(f)



# Sort by date for time-series split
df_sorted = df.sort_values("date")

X = df_sorted.drop(columns=['next_match_runs'])
y = df_sorted['next_match_runs']

# 80% train, 20% test
split_index = int(len(df_sorted) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]



baseline_pred = X_test['avg_runs_last_5']
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print("Baseline RMSE:", baseline_rmse)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)



rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_processed, y_train)

rf_pred = rf.predict(X_test_processed)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest RMSE:", rf_rmse)
print("Random Forest MAE:", rf_mae)
print("Random Forest R2:", rf_r2)

joblib.dump(rf, "models/rf_model.joblib")


from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train_processed, y_train)

xgb_pred = xgb.predict(X_test_processed)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

print("XGBoost RMSE:", xgb_rmse)
print("XGBoost MAE:", xgb_mae)
print("XGBoost R2:", xgb_r2)

joblib.dump(xgb, "models/xgb_model.joblib")


param_grid = {
    'max_depth': [4, 6],
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

grid.fit(X_train_processed, y_train)

print("Best P" \
"" \
"arams:", grid.best_params_)


import shap

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_train_processed[:300])

shap.summary_plot(shap_values, X_train_processed[:300], show=False)



print("\nFINAL COMPARISON")
print("Baseline RMSE:", baseline_rmse)
print("RF RMSE:", rf_rmse)
print("XGB RMSE:", xgb_rmse)
