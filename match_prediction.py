import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

matches = pd.read_csv("data/matches.csv")

matches = matches.dropna(subset=["winner"])

X = matches[["team1","team2","venue"]]
y = matches["winner"]

X = pd.get_dummies(X)

model = RandomForestClassifier()
model.fit(X,y)

pickle.dump(model,open("models/match_model.pkl","wb"))
print("Match model saved")
