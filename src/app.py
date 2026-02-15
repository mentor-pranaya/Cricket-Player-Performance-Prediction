
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Cricket Analytics", layout="wide")

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv(
        r"D:\Infosys_AI_project\ai_cricket_01\notebooks\innings_df.csv",
        low_memory=False
    )
@st.cache_data
def loaded_data():
    return pd.read_csv(
        r"D:\Infosys_AI_project\ai_cricket_01\notebooks\final_df.csv",
        low_memory=False
    )

@st.cache_resource
def load_model():
    return joblib.load(
        r"D:\Infosys_AI_project\ai_cricket_01\notebooks\innings_lightgbm_model.joblib"
    )

final_df = load_data()
ball_df = loaded_data()
model = load_model()

# ==============================
# FEATURE COLUMNS (MATCH DATASET)
# ==============================

feature_cols = [
    'career_avg',
    'recent_3_avg',
    'recent_5_avg',
    'recent_10_avg',
    'strike_rate',
    'boundary_ratio',
    'consistency_score',
    'momentum',
    'venue_adjusted_runs',
    'pvt_avg',
    'experience_log'
]

# ==============================
# TITLE
# ==============================

st.title("🏏 AI-Powered Cricket Analytics Dashboard")

menu = st.sidebar.radio(
    "Select Mode",
    ["Player Performance Predictor",
     "Player vs Player Comparison",
     "Match Simulation",
     "Tournament Analytics"]
)

# =========================================================
# 1️⃣ PLAYER PERFORMANCE PREDICTOR
# =========================================================

if menu == "Player Performance Predictor":

    st.header("🔥 Player Run Prediction")

    player = st.selectbox("Select Player", final_df['batter'].unique())

    player_row = final_df[final_df['batter'] == player].iloc[-1:]

    input_df = player_row[feature_cols]

    prediction = model.predict(input_df)[0]

    st.subheader(f"Predicted Runs for {player}: {round(prediction, 2)}")

    # ===== SHAP Explanation =====
    st.subheader("🔍 Model Explanation (SHAP)")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

   

  


# =========================================================
# 2️⃣ PLAYER VS PLAYER COMPARISON
# =========================================================

elif menu == "Player vs Player Comparison":

    st.header("⚔ Player vs Player Comparison")

    col1, col2 = st.columns(2)

    with col1:
        player1 = st.selectbox("Select Player 1", final_df['batter'].unique())

    with col2:
        player2 = st.selectbox("Select Player 2", final_df['batter'].unique())

    row1 = final_df[final_df['batter'] == player1].iloc[-1:]
    row2 = final_df[final_df['batter'] == player2].iloc[-1:]

    pred1 = model.predict(row1[feature_cols])[0]
    pred2 = model.predict(row2[feature_cols])[0]

    st.write(f"### {player1} Predicted Runs: {round(pred1,2)}")
    st.write(f"### {player2} Predicted Runs: {round(pred2,2)}")

    if pred1 > pred2:
        st.success(f"{player1} is expected to perform better!")
    else:
        st.success(f"{player2} is expected to perform better!")

# =========================================================
# 3️⃣ MATCH SIMULATION
# =========================================================

elif menu == "Match Simulation":

    st.header("🎮 Match Simulation Mode")

    players_list = list(final_df['batter'].unique())

    teamA = st.multiselect("Select Team A Players", players_list, max_selections=11)
    teamB = st.multiselect("Select Team B Players", players_list, max_selections=11)

    def simulate_team(team):
        total_score = 0
        for p in team:
            row = final_df[final_df['batter'] == p].iloc[-1:]
            if not row.empty:
                pred = model.predict(row[feature_cols])[0]
                total_score += pred
        return total_score

    if st.button("Simulate Match"):

        scoreA = simulate_team(teamA)
        scoreB = simulate_team(teamB)

        st.write(f"### Team A Score: {round(scoreA)}")
        st.write(f"### Team B Score: {round(scoreB)}")

        if scoreA > scoreB:
            st.success("🏆 Team A Wins!")
        elif scoreB > scoreA:
            st.success("🏆 Team B Wins!")
        else:
            st.warning("🤝 Match Draw!")

# =========================================================
# 4️⃣ TOURNAMENT ANALYTICS
# =========================================================

elif menu == "Tournament Analytics":

    st.header("📊 Tournament Analytics Dashboard")

    st.subheader("Top 10 Run Scorers")

    top_batters = (
        final_df.groupby('batter')['target_next_runs'].sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_batters)

    st.subheader("Average Runs by Venue")
    
    venue_avg = (
        ball_df.groupby('venue')['batter_runs']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(venue_avg)

