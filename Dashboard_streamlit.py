import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# PAGE CONFIG

st.set_page_config(
    page_title="Cricket Player Performance Prediction",
    layout="wide"
)

# CUSTOM CSS

st.markdown("""
<style>
.main {background-color:#0e1117;}
h1 {color:#ffffff;text-align:center;}

.big-card {
background:#1565c0;
padding:30px;
border-radius:18px;
text-align:center;
box-shadow:0 0 15px rgba(21,101,192,0.8);
}

.big-number {
font-size:60px;
font-weight:900;
color:white;
}

.run {color:white;}
.wicket {color:white;}

.stButton>button {
background:linear-gradient(90deg,#00ffcc,#0099ff);
color:black;font-weight:bold;border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# LOAD DATA

ball = pd.read_csv("Datasets/Cleaned_Datasets/ball_cleaned_data.csv")
matches = pd.read_csv("Datasets/Cleaned_Datasets/matches_cleaned_data.csv")

# LOAD MODELS

batsman_model = joblib.load("xgb_model.joblib")
bowler_model  = joblib.load("bowler_rfmodel.joblib")

BAT_FEATURES  = batsman_model.get_booster().feature_names
BOWL_FEATURES = bowler_model.get_booster().feature_names

# SIDEBAR INPUTS

st.sidebar.title("Match Inputs")

batter_list = ["None"] + sorted(ball["batter"].dropna().unique())
bowler_list = ["None"] + sorted(ball["bowler"].dropna().unique())
team_list   = sorted(ball["team_batting"].dropna().unique())
venue_list  = sorted(matches["venue"].dropna().unique())

batter = st.sidebar.selectbox("Select Batsman", batter_list)
bowler = st.sidebar.selectbox("Select Bowler", bowler_list)
bat_team = st.sidebar.selectbox("Batting Team", team_list)
bowl_team = st.sidebar.selectbox("Bowling Team", team_list)
venue = st.sidebar.selectbox("Venue", venue_list)

# HELPERS

def team_avg_runs(team):
    return int(ball[ball["team_batting"]==team]
               .groupby("match_id")["batter_runs"].sum().mean())

def team_avg_wickets(team):
    return int(ball[ball["team_batting"]==team]
               .groupby("match_id")["is_wicket"].sum().mean())

def prepare_batsman_features():
    X = pd.DataFrame(np.zeros((1,len(BAT_FEATURES))), columns=BAT_FEATURES)
    hist = ball[ball["batter"]==batter]
    X["form_runs_last_10"] = hist.groupby("match_id")["batter_runs"].sum().tail(10).mean()
    return X

def prepare_bowler_features():
    X = pd.DataFrame(np.zeros((1,len(BOWL_FEATURES))), columns=BOWL_FEATURES)
    hist = ball[ball["bowler"]==bowler]
    X["form_wickets_last_10"] = hist.groupby("match_id")["is_wicket"].sum().tail(10).mean()
    return X

# MAIN DASHBOARD

st.title("🏏 Cricket Player Performance Prediction")

if st.sidebar.button("Predict Performance"):
    
    # PREDICTIONS

    bat_pred = team_avg_runs(bat_team) if batter=="None" else \
               int(batsman_model.predict(prepare_batsman_features())[0])

    bowl_pred = team_avg_wickets(bat_team) if bowler=="None" else \
                int(bowler_model.predict(prepare_bowler_features())[0])

    c1,c2 = st.columns(2)

    c1.markdown(f"""
    <div class="big-card">
        <div style="color:white;">Predicted Runs</div>
        <div class="big-number run">{bat_pred}</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="big-card">
        <div style="color:white;">Predicted Wickets</div>
        <div class="big-number wicket">{bowl_pred}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # PLAYER MODE

    if batter!="None" and bowler!="None":

        col1,col2 = st.columns(2)

        bat_last = ball[ball["batter"]==batter] \
            .groupby("match_id")["batter_runs"].sum().tail(10)

        col1.subheader("Batsman Last 10 Matches")
        col1.line_chart(bat_last, height=200)

        bowl_last = ball[ball["bowler"]==bowler] \
            .groupby("match_id")["is_wicket"].sum().tail(10)

        col2.subheader("Bowler Last 10 Matches")
        col2.line_chart(bowl_last, height=200)

        st.markdown("---")

        st.subheader("Why This Prediction (SHAP)")

        explainer = shap.TreeExplainer(batsman_model)
        shap_vals = explainer.shap_values(prepare_batsman_features())

        shap_df = pd.DataFrame({
            "Feature": BAT_FEATURES,
            "Impact": np.abs(shap_vals[0])
        }).sort_values(by="Impact",ascending=False).head(6)

        #  SMALLER SHAP SIZE
        fig,ax = plt.subplots(figsize=(3,2))
        ax.barh(shap_df["Feature"],shap_df["Impact"])
        ax.invert_yaxis()
        st.pyplot(fig)

    # TEAM MODE

    else:

        st.subheader("Match Summary")

        summary = pd.DataFrame({
            "Team":[bat_team,bowl_team],
            "Avg Runs Per Match":[team_avg_runs(bat_team),
                                  team_avg_runs(bowl_team)],
            "Avg Wickets Per Match":[team_avg_wickets(bat_team),
                                     team_avg_wickets(bowl_team)]
        })

        st.table(summary)