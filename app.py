import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Cricket  Performance Predictor",
    layout="wide"
)

st.title("🏏  Player Performance Predictor")
st.markdown("###  Cricket Analytics Dashboard")
st.markdown("---")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    runs_model = pickle.load(open("runs_model.pkl", "rb"))
    wickets_model = pickle.load(open("wickets_model.pkl", "rb"))
    boundaries_model = pickle.load(open("boundaries_model.pkl", "rb"))
    return runs_model, wickets_model, boundaries_model

runs_model, wickets_model, boundaries_model = load_models()

# --------------------------------------------------
# AUTO ALIGN FEATURES FUNCTION
# --------------------------------------------------
def align_features(input_df, model):
    if hasattr(model, "feature_names_in_"):
        required_cols = model.feature_names_in_
        input_df = input_df.reindex(columns=required_cols, fill_value=0)
    return input_df

# --------------------------------------------------
# PLAYER ROLE MAPPING
# --------------------------------------------------
player_roles = {
    "Virat Kohli": "Batsman",
    "Rohit Sharma": "Batsman",
    "MS Dhoni": "Batsman",
    "Jasprit Bumrah": "Bowler",
    "Rashid Khan": "Bowler"
}

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("Match Configuration")

player = st.sidebar.selectbox("Select Player", list(player_roles.keys()))

opponent_team = st.sidebar.selectbox(
    "Select Opponent Team",
    ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore"]
)

venue = st.sidebar.selectbox(
    "Select Venue",
    ["MA Chidambaram Stadium, Chepauk", "Wankhede Stadium", "Eden Gardens"]
)

role = player_roles[player]

st.markdown(f"### 🧾 Role Detected: **{role}**")
st.markdown("---")

# --------------------------------------------------
# SIMULATED LAST 10 MATCH PERFORMANCE (Visualization)
# --------------------------------------------------
np.random.seed(abs(hash(player)) % 1000)

if role == "Batsman":
    last10_values = np.random.randint(20, 90, 10)
    perf_label = "Runs"
else:
    last10_values = np.round(np.random.uniform(0, 4, 10), 1)
    perf_label = "Wickets"

last10_df = pd.DataFrame({
    "Match": [f"M{i}" for i in range(1, 11)],
    perf_label: last10_values
})

col1, col2 = st.columns([2, 1])

with col1:
    fig = px.line(
        last10_df,
        x="Match",
        y=perf_label,
        markers=True,
        title=f"{player} - Last 10 Matches"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# CREATE FEATURES
# --------------------------------------------------
last_5_avg = np.mean(last10_values[-5:])
last_5_sr = np.random.randint(110, 160)
boundary_rate = np.random.uniform(0.3, 0.8)
current_run_rate = np.random.uniform(6, 10)
match_phase = "middle"

input_dict = {
    "last_5_runs": [last_5_avg if role == "Batsman" else 0],
    "last_5_wickets": [last_5_avg if role == "Bowler" else 0],
    "boundary_rate": [boundary_rate],
    "last_5_sr": [last_5_sr],
    "current_run_rate": [current_run_rate],
    "match_phase": [match_phase],
    "venue": [venue],
    "batting_team": ["Selected Team"],
    "bowling_team": [opponent_team]
}

input_df = pd.DataFrame(input_dict)
input_df = pd.get_dummies(input_df)

# --------------------------------------------------
# PREDICTION PANEL
# --------------------------------------------------
with col2:
    st.subheader("🎯 Prediction Panel")

    if st.button("Predict Performance"):

        if role == "Batsman":

            input_runs = align_features(input_df.copy(), runs_model)
            input_boundaries = align_features(input_df.copy(), boundaries_model)

            runs_pred = max(0, int(round(runs_model.predict(input_runs)[0])))
            boundaries_pred = max(0, int(round(boundaries_model.predict(input_boundaries)[0])))

            st.metric("Predicted Runs", runs_pred)
            st.metric("Predicted Boundaries", boundaries_pred)

            # SHAP
            st.markdown("#### 🔍 Feature Impact (Runs)")
            try:
                explainer = shap.Explainer(runs_model)
                shap_values = explainer(input_runs)

                fig_shap, ax = plt.subplots()
                shap.plots.bar(shap_values, show=False)
                st.pyplot(fig_shap)
            except:
                st.warning("SHAP visualization not available.")

        else:

            input_wickets = align_features(input_df.copy(), wickets_model)
            wickets_pred = max(0, int(round(wickets_model.predict(input_wickets)[0])))

            st.metric("Predicted Wickets", wickets_pred)

            # SHAP
            st.markdown("#### 🔍 Feature Impact (Wickets)")
            try:
                explainer = shap.Explainer(wickets_model)
                shap_values = explainer(input_wickets)

                fig_shap, ax = plt.subplots()
                shap.plots.bar(shap_values, show=False)
                st.pyplot(fig_shap)
            except:
                st.warning("SHAP visualization not available.")

# --------------------------------------------------
# PERFORMANCE SUMMARY
# --------------------------------------------------
st.markdown("---")
st.subheader("📊 Performance Summary")

avg_perf = round(np.mean(last10_values), 2)
best_perf = np.max(last10_values)
low_perf = np.min(last10_values)

c1, c2, c3 = st.columns(3)

c1.metric("Average (Last 10)", avg_perf)
c2.metric("Best Performance", best_perf)
c3.metric("Lowest Performance", low_perf)

