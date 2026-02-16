import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="IPL Player Performance Predictor",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Player Performance Predictor")

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

@st.cache_resource
def load_files():
    batsman_model = joblib.load("batsman_full_pipeline.pkl")
    bowler_model = joblib.load("bowler_full_pipeline.pkl")

    batsman_data = pd.read_csv("batsman_match_final_stage2.csv")
    bowler_data = pd.read_csv("bowler_match_final_stage2.csv")

    batsman_data["season"] = pd.to_numeric(batsman_data["season"], errors="coerce")
    bowler_data["season"] = pd.to_numeric(bowler_data["season"], errors="coerce")

    batsman_data = batsman_data.fillna(0)
    bowler_data = bowler_data.fillna(0)

    return batsman_model, bowler_model, batsman_data, bowler_data

batsman_model, bowler_model, batsman_data, bowler_data = load_files()

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

role = st.sidebar.radio("Select Role", ["Batsman", "Bowler"])

if role == "Batsman":
    df = batsman_data
    player_col = "batsman"
else:
    df = bowler_data
    player_col = "bowler"

player = st.sidebar.selectbox("Select Player", sorted(df[player_col].unique()))

if role == "Batsman":
    opponent = st.sidebar.selectbox("Select Opponent", sorted(df["bowling_team"].unique()))
else:
    opponent = st.sidebar.selectbox("Select Opponent", sorted(df["batting_team"].unique()))

venue = st.sidebar.selectbox("Select Venue", sorted(df["venue"].unique()))

# --------------------------------------------------
# SAFE FEATURE ALIGNMENT FUNCTION
# --------------------------------------------------

def align_with_pipeline(input_df, pipeline):

    # Extract expected feature order
    expected_features = pipeline.named_steps["preprocessing"].feature_names_in_

    # Reorder
    input_df = input_df[expected_features].copy()

    # Convert categorical to plain python strings
    for col in input_df.columns:
        if input_df[col].dtype == object or input_df[col].dtype == "string":
            input_df[col] = input_df[col].astype(str)

    # Convert everything else strictly to float64
    for col in input_df.columns:
        if col not in ["venue", "city", "batting_team", "bowling_team"]:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").astype(np.float64)

    # Replace any remaining NaN
    input_df = input_df.replace([np.inf, -np.inf], 0)
    input_df = input_df.fillna(0)

    # FINAL CRITICAL STEP:
    # Remove pandas extension types entirely
    input_df = pd.DataFrame(
        input_df.values,
        columns=input_df.columns
    )

    return input_df


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def create_batsman_features():

    pdf = df[df["batsman"] == player].sort_values("season")

    if len(pdf) < 5:
        return None

    latest = pdf.iloc[-1]

    total_balls = pdf["balls_faced"].sum()
    boundary_rate = ((pdf["fours"].sum() + pdf["sixes"].sum()) / total_balls) if total_balls > 0 else 0

    features = {
        "season": int(latest["season"]) + 1,
        "venue": venue,
        "city": str(latest["city"]),
        "batting_team": str(latest["batting_team"]),
        "bowling_team": str(opponent),
        "balls_faced": float(pdf["balls_faced"].mean()),
        "fours": float(pdf["fours"].mean()),
        "sixes": float(pdf["sixes"].mean()),
        "strike_rate": float(pdf["strike_rate"].mean()),
        "avg_runs_last_5": float(pdf.tail(5)["runs"].mean()),
        "avg_runs_last_10": float(pdf.tail(10)["runs"].mean()),
        "avg_runs_at_venue": float(pdf[pdf["venue"]==venue]["runs"].mean()) if len(pdf[pdf["venue"]==venue])>0 else 0.0,
        "matches_at_venue": int(len(pdf[pdf["venue"]==venue])),
        "matches_played": int(len(pdf)),
        "career_avg_runs": float(pdf["runs"].mean()),
        "career_avg_balls": float(pdf["balls_faced"].mean()),
        "boundary_rate": float(boundary_rate),
        "recent_form_indicator": float(pdf.tail(5)["runs"].mean()),
        "experience_level": int(1 if len(pdf) > 50 else 0)
    }

    return pd.DataFrame([features])


def create_bowler_features():

    pdf = df[df["bowler"] == player].sort_values("season")

    if len(pdf) < 5:
        return None

    latest = pdf.iloc[-1]

    discipline = 1 / (pdf["wides"].mean() + pdf["no_balls"].mean() + 1)

    features = {
        "season": int(latest["season"]) + 1,
        "venue": venue,
        "city": str(latest["city"]),
        "bowling_team": str(latest["bowling_team"]),
        "batting_team": str(opponent),
        "runs_conceded": float(pdf["runs_conceded"].mean()),
        "balls_bowled": float(pdf["balls_bowled"].mean()),
        "wides": float(pdf["wides"].mean()),
        "no_balls": float(pdf["no_balls"].mean()),
        "overs": float(pdf["overs"].mean()),
        "economy": float(pdf["economy"].mean()),
        "avg_wkts_last_5": float(pdf.tail(5)["wicket"].mean()),
        "avg_wkts_last_10": float(pdf.tail(10)["wicket"].mean()),
        "avg_wkts_at_venue": float(pdf[pdf["venue"]==venue]["wicket"].mean()) if len(pdf[pdf["venue"]==venue])>0 else 0.0,
        "matches_at_venue": int(len(pdf[pdf["venue"]==venue])),
        "matches_played": int(len(pdf)),
        "career_avg_wickets": float(pdf["wicket"].mean()),
        "career_avg_runs_conceded": float(pdf["runs_conceded"].mean()),
        "career_avg_overs": float(pdf["overs"].mean()),
        "discipline_score": float(discipline),
        "recent_vs_career_form": float(pdf.tail(5)["wicket"].mean() - pdf["wicket"].mean())
    }

    return pd.DataFrame([features])

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict"):

    try:
        if role == "Batsman":
            input_df = create_batsman_features()
            if input_df is None:
                st.warning("Not enough historical data.")
            else:
                input_df = align_with_pipeline(input_df, batsman_model)
                pred = batsman_model.predict(input_df)[0]
                st.success(f"🏏 Predicted Runs: {round(float(pred),2)}")

        else:
            input_df = create_bowler_features()
            if input_df is None:
                st.warning("Not enough historical data.")
            else:
                input_df = align_with_pipeline(input_df, bowler_model)
                pred = bowler_model.predict(input_df)[0]
                st.success(f"🎯 Predicted Wickets: {round(float(pred),2)}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
