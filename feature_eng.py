import pandas as pd


# STEP 1: Load the merged IPL dataset
# This dataset contains ball-by-ball data merged with match details
df = pd.read_csv("data/cleaned/ipl_merged_data.csv")

# Convert the 'date' column to datetime format for proper sorting
df['date'] = pd.to_datetime(df['date'])

# Sort data by batsman and date to maintain chronological order
df = df.sort_values(['batsman', 'date'])

# STEP 2: Aggregate ball-by-ball data to player-match level
# Grouping data so that each row represents one batsman's performance in one match
player_match = df.groupby(
    ['match_id', 'date', 'batsman', 'venue', 'batting_team', 'bowling_team']
).agg(
    # Total runs scored by the batsman in that match
    runs_scored=('batsman_runs', 'sum'),
    
    # Total balls faced by the batsman in that match
    balls_faced=('batsman_runs', 'count')
).reset_index()


# STEP 3: Calculate strike rate
# Strike rate = (runs scored / balls faced) * 100
player_match['strike_rate'] = (
    player_match['runs_scored'] / player_match['balls_faced']
) * 100

# Sort again to ensure rolling calculations are correct
player_match = player_match.sort_values(['batsman', 'date'])

# STEP 4: Create recent form features (rolling averages)
# Average runs scored by the batsman in the last 3 matches
player_match['avg_runs_last_3'] = (
    player_match.groupby('batsman')['runs_scored']
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Average runs scored by the batsman in the last 5 matches
player_match['avg_runs_last_5'] = (
    player_match.groupby('batsman')['runs_scored']
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# STEP 5: Venue-wise performance feature
# Calculate average runs scored by a batsman at each venue
venue_avg = player_match.groupby(
    ['batsman', 'venue']
)['runs_scored'].mean().reset_index(name='venue_avg_runs')

# Merge venue-wise average runs back into main dataset
player_match = player_match.merge(
    venue_avg,
    on=['batsman', 'venue'],
    how='left'
)

# STEP 6: Career-level features
# Calculate total career runs and career average runs for each batsman
career = player_match.groupby('batsman').agg(
    career_runs=('runs_scored', 'sum'),
    career_avg=('runs_scored', 'mean')
).reset_index()

# Merge career statistics into the main dataset
player_match = player_match.merge(
    career,
    on='batsman',
    how='left'
)

# STEP 7: Create target variable (next match runs)
# Shift runs_scored column to get runs scored in the next match
# This will be used as the prediction target
player_match['next_match_runs'] = (
    player_match.groupby('batsman')['runs_scored']
    .shift(-1)
)

# Remove rows where next match data is not available
player_match = player_match.dropna()

# STEP 8: Save the final feature-engineered dataset
player_match.to_csv("data/cleaned/dataset.csv", index=False)

print(" Feature engineering completed")
