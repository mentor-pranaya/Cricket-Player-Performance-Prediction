import pandas as pd

deliveries_df = pd.read_csv("deliveries.csv")
matches_df = pd.read_csv("matches.csv")

print(deliveries_df.shape)
print(matches_df.shape)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")


# ----------------------------
# Column names
# ----------------------------
print("\nDeliveries columns:")
print(deliveries_df.columns)

print("\nMatches columns:")
print(matches_df.columns)

# ----------------------------
# Basic information
# ----------------------------
print("\nDeliveries info:")
print(deliveries_df.info())

print("\nMatches info:")
print(matches_df.info())

# ----------------------------
# Missing values
# ----------------------------
print("\nMissing values in deliveries:")
print(deliveries_df.isnull().sum())

print("\nMissing values in matches:")
print(matches_df.isnull().sum())

# ----------------------------
# Matches per season
# ----------------------------
plt.figure(figsize=(8,4))
matches_df['season'].value_counts().sort_index().plot(kind='bar')
plt.title("Matches per Season")
plt.xlabel("Season")
plt.ylabel("No of Matches")
plt.tight_layout()
plt.savefig("matches_per_season.png")
plt.close()



# ----------------------------
# Top 10 venues
# ----------------------------
plt.figure(figsize=(10,4))
matches_df['venue'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Venues")
plt.tight_layout()
plt.savefig("top_venues.png")
plt.close()


# ----------------------------
# Toss decision distribution
# ----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='toss_decision', data=matches_df)
plt.title("Toss Decision Distribution")
plt.tight_layout()
plt.savefig("toss_decision.png")
plt.close()


# ----------------------------
# Top 10 teams by wins
# ----------------------------
plt.figure(figsize=(8,4))
matches_df['winner'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Teams by Wins")
plt.tight_layout()
plt.savefig("top_teams.png")
plt.close()


# ----------------------------
# Runs per ball
# ----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='batsman_runs', data=deliveries_df)
plt.title("Runs per ball")
plt.tight_layout()
plt.savefig("runs_per_ball.png")
plt.close()

# ----------------------------
# Total runs per delivery
# ----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='total_runs', data=deliveries_df)
plt.title("Total runs per delivery")
plt.tight_layout()
plt.savefig("total_runs_per_ball.png")
plt.close()


# ----------------------------
# Top 10 batsmen by total runs
# ----------------------------
top_batsmen = (
    deliveries_df
    .groupby('batter')['batsman_runs']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(8,4))
top_batsmen.plot(kind='bar')
plt.title("Top 10 Batsmen by Runs")
plt.tight_layout()
plt.savefig("top_batsmen.png")
plt.close()


# ----------------------------
# Top 10 bowlers by wickets
# ----------------------------
wickets_df = deliveries_df[deliveries_df['dismissal_kind'].notna()]

top_bowlers = wickets_df['bowler'].value_counts().head(10)

plt.figure(figsize=(8,4))
top_bowlers.plot(kind='bar')
plt.title("Top 10 Bowlers by Wickets")
plt.tight_layout()
plt.savefig("top_bowlers.png")
plt.close()

