import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="IPL Player Performance Predictor", layout="wide")

st.title("🏏 IPL Player Performance Prediction Dashboard")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    batsman = pd.read_csv("final_batsman_features.csv")
    bowler = pd.read_csv("final_bowler_features.csv")
    batsman["match_date"] = pd.to_datetime(batsman["match_date"])
    bowler["match_date"] = pd.to_datetime(bowler["match_date"])
    return batsman, bowler

@st.cache_resource
def load_models():
    xgb_bat = pickle.load(open("best_batsman_model.pkl","rb"))
    xgb_bowl = pickle.load(open("best_bowler_model.pkl","rb"))
    return xgb_bat, xgb_bowl

batsman, bowler = load_data()
xgb_bat, xgb_bowl = load_models()

batsman_features = [
    "career_avg_runs","venue_avg_runs","opponent_avg_runs",
    "prev_10_avg_runs","strike_rate","fours","sixes"
]

bowler_features = [
    "career_avg_wkts","venue_avg_wkts","opponent_avg_wkts",
    "prev_10_avg_wkts","economy"
]

tab1, tab2 = st.tabs(["Batsman Runs Prediction", "Bowler Wickets Prediction"])

# ============================================================
# ======================= BATSMAN TAB ========================
# ============================================================

with tab1:
    st.header("Predict Runs for a Batsman")

    bat_player = st.selectbox(
        "Select Batsman",
        sorted(batsman["batsman"].unique()),
        key="bat_player"
    )

    bat_opponent = st.selectbox(
        "Select Opponent Team",
        sorted(batsman["bowling_team"].unique()),
        key="bat_opponent"
    )

    bat_venue = st.selectbox(
        "Select Venue",
        sorted(batsman["venue"].unique()),
        key="bat_venue"
    )

    filtered = batsman[
        (batsman["batsman"] == bat_player) &
        (batsman["bowling_team"] == bat_opponent) &
        (batsman["venue"] == bat_venue)
    ]

    if not filtered.empty:
        latest = filtered.sort_values("match_date").iloc[-1]
        input_df = pd.DataFrame([latest[batsman_features]])

        prediction = xgb_bat.predict(input_df)[0]

        st.subheader(f"🎯 Predicted Runs: {round(prediction,2)}")

        st.subheader("📈 Recent Form (Last 10 Matches)")
        recent = batsman[batsman["batsman"] == bat_player] \
            .sort_values("match_date") \
            .tail(10)
        st.line_chart(recent.set_index("match_date")["runs_scored"])

        st.subheader("🔍 SHAP Feature Importance")

        explainer = shap.TreeExplainer(xgb_bat)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig)

    else:
        st.warning("No historical data available for this combination.")

# ============================================================
# ======================= BOWLER TAB =========================
# ============================================================

with tab2:
    st.header("Predict Wickets for a Bowler")

    bowl_player = st.selectbox(
        "Select Bowler",
        sorted(bowler["bowler"].unique()),
        key="bowl_player"
    )

    bowl_opponent = st.selectbox(
        "Select Opponent Team",
        sorted(bowler["batting_team"].unique()),
        key="bowl_opponent"
    )

    bowl_venue = st.selectbox(
        "Select Venue",
        sorted(bowler["venue"].unique()),
        key="bowl_venue"
    )

    filtered = bowler[
        (bowler["bowler"] == bowl_player) &
        (bowler["batting_team"] == bowl_opponent) &
        (bowler["venue"] == bowl_venue)
    ]

    if not filtered.empty:
        latest = filtered.sort_values("match_date").iloc[-1]
        input_df = pd.DataFrame([latest[bowler_features]])

        prediction = xgb_bowl.predict(input_df)[0]

        st.subheader(f"🎯 Predicted Wickets: {round(prediction,2)}")

        st.subheader("📈 Recent Form (Last 10 Matches)")
        recent = bowler[bowler["bowler"] == bowl_player] \
            .sort_values("match_date") \
            .tail(10)
        st.line_chart(recent.set_index("match_date")["wickets"])

        st.subheader("🔍 SHAP Feature Importance")

        explainer = shap.TreeExplainer(xgb_bowl)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig)

    else:
        st.warning("No historical data available for this combination.")
