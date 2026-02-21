import pandas as pd

# ---------------------------
# Load datasets
# ---------------------------

deliveries_df = pd.read_csv("deliveries.csv")
matches_df = pd.read_csv("matches.csv")

print("Deliveries shape:", deliveries_df.shape)
print("Matches shape:", matches_df.shape)

# ---------------------------
# Keep only needed columns from matches
# ---------------------------

matches_small = matches_df[['id', 'season']]

# ---------------------------
# Merge deliveries with matches
# ---------------------------

merged_df = deliveries_df.merge(
    matches_small,
    left_on='match_id',
    right_on='id',
    how='left'
)

print("Merged data shape:", merged_df.shape)

# ---------------------------
# Create player–match level dataset
# ---------------------------

player_match_df = (
    merged_df
    .groupby(['season', 'match_id', 'batter'])
    .agg(
        runs=('batsman_runs', 'sum'),
        balls=('ball', 'count')
    )
    .reset_index()
)

print(player_match_df.head())

# ---------------------------
# Save the dataset
# ---------------------------

player_match_df.to_csv(
    "player_match_level.csv",
    index=False
)

print("player_match_level.csv created successfully")
