import pandas as pd

# -----------------------------------
# Step 1 : Load player–match dataset
# -----------------------------------

df = pd.read_csv("player_match_level.csv")

print(df.head())
print(df.shape)

# -----------------------------------
# Step 2 : Sort properly (important)
# -----------------------------------

df = df.sort_values(
    ['batter', 'season', 'match_id']
).reset_index(drop=True)

# -----------------------------------
# Step 3 : Create strike rate
# -----------------------------------

df['strike_rate'] = (df['runs'] / df['balls']) * 100

# -----------------------------------
# Step 4 : Matches played till now
# -----------------------------------

df['matches_played'] = (
    df.groupby('batter')
      .cumcount()
)

# -----------------------------------
# Step 5 : Career average runs till previous match
# -----------------------------------

df['career_avg_runs'] = (
    df.groupby('batter')['runs']
      .expanding()
      .mean()
      .shift()
      .reset_index(level=0, drop=True)
)

# -----------------------------------
# Step 6 : Rolling average of runs (last 5 matches)
# -----------------------------------

df['avg_runs_last_5'] = (
    df.groupby('batter')['runs']
      .shift(1)
      .rolling(5)
      .mean()
)

# -----------------------------------
# Step 7 : Rolling average strike rate (last 5 matches)
# -----------------------------------

df['avg_sr_last_5'] = (
    df.groupby('batter')['strike_rate']
      .shift(1)
      .rolling(5)
      .mean()
)

# -----------------------------------
# Step 8 : Remove rows where features are missing
# -----------------------------------

feature_cols = [
    'matches_played',
    'career_avg_runs',
    'avg_runs_last_5',
    'avg_sr_last_5'
]

df_final = df.dropna(subset=feature_cols).copy()

# -----------------------------------
# Step 9 : Select final columns for ML
# -----------------------------------

final_df = df_final[
    [
        'season',
        'batter',
        'matches_played',
        'avg_runs_last_5',
        'avg_sr_last_5',
        'career_avg_runs',
        'runs'
    ]
]

print(final_df.head())
print(final_df.shape)

# -----------------------------------
# Step 10 : Save feature dataset
# -----------------------------------

final_df.to_csv(
    "batsman_features.csv",
    index=False
)

print("batsman_features.csv created successfully")
