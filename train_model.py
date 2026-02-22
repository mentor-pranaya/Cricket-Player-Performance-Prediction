import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

batsman = pd.read_csv("data/final_batsman.csv")
bowler = pd.read_csv("data/final_bowler.csv")

bat_features = ["career_avg_runs","prev_10_avg_runs","strike_rate","fours","sixes"]
bowl_features = ["career_avg_wkts","prev_10_avg_wkts","economy"]

X_bat = batsman[bat_features]
y_bat = batsman["runs_scored"]

X_bowl = bowler[bowl_features]
y_bowl = bowler["wickets"]

# ---------- BAT MODEL TUNING ----------
params = {
    "max_depth":[3,5,7],
    "n_estimators":[100,200]
}

grid_bat = GridSearchCV(XGBRegressor(),params,cv=3)
grid_bat.fit(X_bat,y_bat)

bat_model = grid_bat.best_estimator_

# ---------- BOWL MODEL ----------
bowl_model = XGBRegressor()
bowl_model.fit(X_bowl,y_bowl)

pickle.dump(bat_model,open("models/bat_model.pkl","wb"))
pickle.dump(bowl_model,open("models/bowl_model.pkl","wb"))

print("Models saved with tuning")
