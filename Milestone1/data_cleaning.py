import pandas as pd

# ==============================
# Load raw datasets
# ==============================
deliveries = pd.read_csv("data/deliveries.csv")
matches = pd.read_csv("data/matches.csv")

# ==============================
# Inspect basic info
# ==============================
print("Deliveries shape:", deliveries.shape)
print("Matches shape:", matches.shape)

# ==============================
# Standardize column names
# ==============================
# matches dataset uses 'id' instead of 'match_id'
matches = matches.rename(columns={"id": "match_id"})

# ==============================
# Select required columns
# ==============================
deliveries_cleaned = deliveries[
    ["match_id", "inning", "batter", "bowler", "batsman_runs", "is_wicket"]
]

matches_cleaned = matches[
    ["match_id", "season", "venue", "team1", "team2", "winner"]
]

# ==============================
# Handle missing values
# ==============================
deliveries_cleaned = deliveries_cleaned.dropna()
matches_cleaned = matches_cleaned.dropna(subset=["venue"])

# ==============================
# Save cleaned datasets
# ==============================
deliveries_cleaned.to_csv("data/deliveries_cleaned.csv", index=False)
matches_cleaned.to_csv("data/matches_cleaned.csv", index=False)

print("✅ Data cleaning completed successfully.")
