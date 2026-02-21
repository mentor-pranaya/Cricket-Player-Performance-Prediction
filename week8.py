import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="IPL Player Performance Prediction", layout="wide")

st.title("🏏 IPL Player Performance Prediction")

# -------------------------------------------------
# Load data
# -------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("lgb_model.pkl")

@st.cache_data
def load_features():
    return pd.read_csv("train_features.csv")

@st.cache_data
def load_player_match():
    return pd.read_csv("player_match_level.csv")

@st.cache_data
def load_matches():
    return pd.read_csv("matches.csv")

model = load_model()
explainer = shap.TreeExplainer(model)

features_df = load_features()
player_df = load_player_match()
matches_df = load_matches()

# -------------------------------------------------
# Merge player + match info to get venue and teams
# -------------------------------------------------

meta = player_df.merge(
    matches_df[["id", "venue", "team1", "team2"]],
    left_on="match_id",
    right_on="id",
    how="left"
)

# -------------------------------------------------
# Sidebar inputs
# -------------------------------------------------

st.sidebar.header("Match Inputs")

players = sorted(meta["batter"].dropna().unique())
player = st.sidebar.selectbox("Select Player", players)

player_rows = meta[meta["batter"] == player]

teams = pd.concat(
    [player_rows["team1"], player_rows["team2"]]
).dropna().unique()

opponent = st.sidebar.selectbox(
    "Select Opponent Team",
    sorted(teams)
)


if "venue" in player_rows.columns:
    venue_col = "venue"
elif "venue_x" in player_rows.columns:
    venue_col = "venue_x"
elif "venue_y" in player_rows.columns:
    venue_col = "venue_y"
else:
    st.error("Venue column not found after merge")
    st.stop()

venues = sorted(player_rows[venue_col].dropna().unique())


venue = st.sidebar.selectbox("Select Venue", venues)

# -------------------------------------------------
# Filter rows for scenario
# -------------------------------------------------

filtered = player_rows[
    ((player_rows["team1"] == opponent) |
     (player_rows["team2"] == opponent)) &
    (player_rows[venue_col] == venue)
]


st.sidebar.write("Matching historical rows:", len(filtered))
# -------------------------------------------------
# Predict
# -------------------------------------------------

if st.sidebar.button("Predict Performance"):

    if len(filtered) > 0:
        row = filtered.sort_values("match_id").iloc[-1]
        source_note = "Using same opponent & venue history"
    else:
        row = player_rows.sort_values("match_id").iloc[-1]
        source_note = "No exact opponent/venue match. Using player's latest match."

    match_id = row["match_id"]

    row_feat = features_df[
        features_df["batter"] == player
    ].sort_values("season").tail(1)

    if row_feat.empty:
        st.error("Feature row not found for this player.")
        st.stop()

    X = row_feat.drop(
        columns=["runs", "batter", "season"],
        errors="ignore"
    )

    X = X.select_dtypes(include=["int64", "float64"])

    pred = model.predict(X)[0]

    # -------------------------------------------------
    # Confidence estimation
    # -------------------------------------------------

    recent_all = features_df[
        features_df["batter"] == player
    ].sort_values("season")

    recent_runs = recent_all["runs"].tail(10)
    std = recent_runs.std()

    if std < 8:
        confidence = "High"
    elif std < 15:
        confidence = "Medium"
    else:
        confidence = "Low"

    # -------------------------------------------------
    # Header
    # -------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Player", player)
    c2.metric("Opponent", opponent)
    c3.metric("Venue", venue)
    c4.metric("Predicted Runs", f"{pred:.0f}", confidence)

    st.info(source_note)

    # -------------------------------------------------
    # Recent performance
    # -------------------------------------------------

    st.subheader("Recent performance")

    recent_plot = recent_all.tail(10).copy()
    recent_plot["match_order"] = range(1, len(recent_plot) + 1)

    if len(recent_plot) > 1:
        st.line_chart(
            recent_plot.set_index("match_order")["runs"]
        )
    else:
        st.warning("Not enough history to show recent form.")

    # -------------------------------------------------
    # SHAP explanation
    # -------------------------------------------------

    st.subheader("Why this prediction? (SHAP explanation)")

    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(7, 4))

    shap.plots.bar(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X.iloc[0],
            feature_names=X.columns
        ),
        show=False
    )

    st.pyplot(fig)

    # -------------------------------------------------
    # Model input features
    # -------------------------------------------------

    st.subheader("Model input features")

    st.dataframe(X.T, use_container_width=True)

else:
    st.info("Select player, opponent and venue, then click Predict Performance.")
