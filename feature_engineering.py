import pandas as pd

batsman = pd.read_csv("data/model_batsman.csv")
bowler = pd.read_csv("data/model_bowler.csv")

# batsman features
batsman["career_avg_runs"] = batsman.groupby("batsman")["runs_scored"].transform("mean")
batsman["prev_10_avg_runs"] = batsman.groupby("batsman")["runs_scored"].transform(lambda x:x.shift().rolling(10).mean())

# bowler features
bowler["career_avg_wkts"] = bowler.groupby("bowler")["wickets"].transform("mean")
bowler["prev_10_avg_wkts"] = bowler.groupby("bowler")["wickets"].transform(lambda x:x.shift().rolling(10).mean())

batsman.fillna(0,inplace=True)
bowler.fillna(0,inplace=True)

batsman.to_csv("data/final_batsman.csv",index=False)
bowler.to_csv("data/final_bowler.csv",index=False)

print("Feature engineering done")
