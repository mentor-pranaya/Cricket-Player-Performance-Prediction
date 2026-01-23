import pandas as pd
import pickle
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------------------------------
# Create folder to store trained models and pipelines
# -----------------------------------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------------------------------
# Load the final dataset (after feature engineering)
# -----------------------------------------------------
df = pd.read_csv("data/cleaned/dataset.csv")

# -----------------------------------------------------
# Separate input features (X) and target is excluded
# -----------------------------------------------------
X = df.drop(columns=["next_match_runs"])

# -----------------------------------------------------
# Numerical columns that need scaling
# -----------------------------------------------------
numeric_features = [
    "runs_scored",
    "balls_faced",
    "strike_rate",
    "avg_runs_last_3",
    "avg_runs_last_5",
    "venue_avg_runs",
    "career_runs",
    "career_avg"
]

# -----------------------------------------------------
# Categorical columns that need encoding
# -----------------------------------------------------
categorical_features = [
    "venue",
    "batting_team",
    "bowling_team",
    "batsman"
]

# -----------------------------------------------------
# Numerical preprocessing:
# StandardScaler is used so all numeric values
# are on a similar scale for the ML model
# -----------------------------------------------------
numeric_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# -----------------------------------------------------
# Categorical preprocessing:
# OneHotEncoder converts text data into numbers
# 'drop=first' avoids dummy variable trap
# 'handle_unknown' prevents errors for new categories
# -----------------------------------------------------
categorical_pipeline = Pipeline(steps=[
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False
    ))
])

# -----------------------------------------------------
# Combine numeric and categorical preprocessing
# using ColumnTransformer
# -----------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)

# -----------------------------------------------------
# Fit the preprocessing pipeline on the input data
# This step learns:
# - mean & std for numerical columns
# - unique categories for categorical columns
# -----------------------------------------------------
preprocessor.fit(X)

# -----------------------------------------------------
# Save the fitted preprocessing pipeline
# so the same transformations can be reused
# during prediction and dashboard deployment
# -----------------------------------------------------
with open("models/feature_pipeline.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print(" Preprocessing pipeline fitted and saved successfully")
