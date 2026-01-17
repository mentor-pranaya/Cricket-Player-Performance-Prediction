import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load the feature-engineered dataset
df = pd.read_csv("data/cleaned/dataset.csv")

# Separate input features (X)
X = df.drop(columns=["next_match_runs"])

# Numerical features
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

# Categorical features
categorical_features = [
    "venue",
    "batting_team",
    "bowling_team",
    "batsman"
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

#  Fit the preprocessor on the data
preprocessor.fit(X)

# Save the fitted preprocessing pipeline
with open("models/feature_pipeline.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("feature_pipeline.pkl saved in models folder")
