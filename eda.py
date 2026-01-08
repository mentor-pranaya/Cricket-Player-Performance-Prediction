import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ===== LOAD CLEANED DATA =====
matches = pd.read_csv("data/cleaned/matches_cleaned.csv")
deliveries = pd.read_csv("data/cleaned/deliveries_cleaned.csv")

# ===== MERGE DATA =====
ipl_data = deliveries.merge(
    matches,
    left_on="match_id",
    right_on="id",
    how="left"
)

print("Merged data shape:", ipl_data.shape)

# ===== RUNS DISTRIBUTION =====
plt.figure(figsize=(8,5))
sns.histplot(ipl_data['total_runs'], bins=20)
plt.title("Distribution of Total Runs per Ball")
plt.show()

# ===== WICKET DISTRIBUTION =====
plt.figure(figsize=(6,4))
sns.countplot(x='is_wicket', data=ipl_data)
plt.title("Wicket Distribution")
plt.show()

# ===== TOP VENUES =====
venue_counts = matches['venue'].value_counts().head(10)

plt.figure(figsize=(10,5))
venue_counts.plot(kind='bar')
plt.title("Top 10 Venues by Matches Played")
plt.show()

# ===== TEAM PERFORMANCE =====
team_wins = matches['winner'].value_counts()

plt.figure(figsize=(10,5))
team_wins.plot(kind='bar')
plt.title("Matches Won by Teams")
plt.show()

# ===== SAVE MERGED DATA =====
ipl_data.to_csv("data/cleaned/ipl_merged_data.csv", index=False)

print("✅ EDA COMPLETED SUCCESSFULLY")
