import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# XGBoost and LightGBM
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# --------------------------------------------------
# Step 1 : Load train and test data
# --------------------------------------------------

train_df = pd.read_csv("train_features.csv")
test_df  = pd.read_csv("test_features.csv")

print(train_df.shape)
print(test_df.shape)


# --------------------------------------------------
# Step 2 : Select features and target
# --------------------------------------------------

feature_cols = [
    'matches_played',
    'avg_runs_last_5',
    'avg_sr_last_5',
    'career_avg_runs'
]

X_train = train_df[feature_cols]
y_train = train_df['runs']

X_test  = test_df[feature_cols]
y_test  = test_df['runs']


# --------------------------------------------------
# Step 3 : Baseline model
# (use avg_runs_last_5 as prediction)
# --------------------------------------------------

baseline_pred = X_test['avg_runs_last_5']

import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

mae  = mean_absolute_error(y_test, baseline_pred)
r2   = r2_score(y_test, baseline_pred)

print("\nBaseline model (avg_runs_last_5)")
print("RMSE :", rmse)
print("MAE  :", mae)
print("R2   :", r2)


# --------------------------------------------------
# Step 4 : Random Forest
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)

print("\nRandom Forest")
print("RMSE :", rf_rmse)
print("MAE  :", rf_mae)
print("R2   :", rf_r2)


# --------------------------------------------------
# Step 5 : XGBoost
# --------------------------------------------------

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

xgb_mae  = mean_absolute_error(y_test, xgb_pred)
xgb_r2   = r2_score(y_test, xgb_pred)

print("\nXGBoost")
print("RMSE :", xgb_rmse)
print("MAE  :", xgb_mae)
print("R2   :", xgb_r2)


# --------------------------------------------------
# Step 6 : LightGBM
# --------------------------------------------------

lgb_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)

lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))

lgb_mae  = mean_absolute_error(y_test, lgb_pred)
lgb_r2   = r2_score(y_test, lgb_pred)

print("\nLightGBM")
print("RMSE :", lgb_rmse)
print("MAE  :", lgb_mae)
print("R2   :", lgb_r2)


# --------------------------------------------------
# Step 7 : Save trained models
# --------------------------------------------------

import joblib

joblib.dump(rf_model,  "rf_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(lgb_model, "lgb_model.pkl")

print("\nModels saved:")
print("rf_model.pkl")
print("xgb_model.pkl")
print("lgb_model.pkl")
