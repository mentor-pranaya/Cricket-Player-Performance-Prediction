import pandas as pd
import numpy as np

deliveries = pd.read_csv("data/deliveries.csv")
# Fix new IPL dataset column names
if "batter" in deliveries.columns:
    deliveries = deliveries.rename(columns={"batter":"batsman"})

if "dismissed_batter" in deliveries.columns:
    deliveries = deliveries.rename(columns={"dismissed_batter":"player_dismissed"})

matches = pd.read_csv("data/matches.csv")

# merge
# find match id column automatically
del_match = [c for c in deliveries.columns if "id" in c.lower()][0]
mat_match = [c for c in matches.columns if "id" in c.lower()][0]

deliveries = deliveries.rename(columns={del_match:"match_id"})
matches = matches.rename(columns={mat_match:"match_id"})

df = deliveries.merge(matches, on="match_id", how="left")


df["match_date"] = pd.to_datetime(df["date"])

# legal ball
df["legal_ball"] = np.where(
    (df.get("wide_runs",0)==0) & (df.get("noball_runs",0)==0),
    1,0
)

# batsman match stats
batsman_match = df.groupby(
    ["match_id","match_date","venue","batsman","batting_team","bowling_team"]
).agg(
    runs_scored=("batsman_runs","sum"),
    balls_faced=("legal_ball","sum"),
    fours=("batsman_runs",lambda x:(x==4).sum()),
    sixes=("batsman_runs",lambda x:(x==6).sum())
).reset_index()

batsman_match["strike_rate"] = (
    batsman_match["runs_scored"]/batsman_match["balls_faced"]*100
)

# bowler stats
bowler_match = df.groupby(
    ["match_id","match_date","venue","bowler","bowling_team","batting_team"]
).agg(
    runs_conceded=("total_runs","sum"),
    legal_balls=("legal_ball","sum"),
    wickets=("player_dismissed","count")
).reset_index()

bowler_match["economy"] = bowler_match["runs_conceded"]/(bowler_match["legal_balls"]/6)

batsman_match.to_csv("data/model_batsman.csv",index=False)
bowler_match.to_csv("data/model_bowler.csv",index=False)

print("Preprocessing done")
