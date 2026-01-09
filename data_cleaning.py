import pandas as pd
import numpy as np

df = pd.read_csv('data/deliveries_initial.csv')

# Standardizing Column Names
df.columns = df.columns.str.strip().str.lower()

# Handling Missing Values
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(0)

cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna("unknown")

# Creating Derived Columns
df["total_runs"] = (
    df["batsman_runs"]
    + df["extras"]
    + df["byes"]
    + df["legbyes"]
    + df["penalty"]
)

df["is_wicket"] = df["player_dismissed"].ne("unknown").astype(int)

# Fixing Data Types
int_cols = [
    "over", "ball", "batsman_runs",
    "total_runs", "is_wicket"
]
for col in int_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)


# Normalizing Text Fields
text_cols = [
    "batsman", "bowler",
    "batting_team", "bowling_team"
]

for col in text_cols:
    df[col] = df[col].str.strip().str.lower()

# Removing Invalid Records
df = df[df["batsman_runs"] >= 0]
df = df[df["total_runs"] >= 0]
df = df[df["over"] <= 20]

# Sorting Chronologically
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values(by=["date", "matchid", "inning", "over", "ball"])

df.to_csv('data/deliveries_cleaned.csv', index=False)

print("Data cleaned successfully")
