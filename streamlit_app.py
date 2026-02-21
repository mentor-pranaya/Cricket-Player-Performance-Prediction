import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Cricket Player Performance Predictor",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 Cricket Player Performance Predictor")
st.write("Predict expected runs using trained ML model")

# -----------------------------
# Load artifacts safely
# -----------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("batsman_xgb_model.pkl", "rb"))

@st.cache_resource
def load_feature_columns():
    return pickle.load(open("feature_columns.pkl", "rb"))

@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_batting_features_v2.csv")

model = load_model()
feature_columns = load_feature_columns()
df = load_dataset()

# -----------------------------
# Detect player column automatically
# -----------------------------
possible_player_cols = [
    "batter",
    "batsman",
    "player",
    "player_name",
    "striker"
]

player_col = None

for col in possible_player_cols:
    if col in df.columns:
        player_col = col
        break

if player_col is None:
    st.error("❌ No player column found in dataset")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# -----------------------------
# Player selector
# -----------------------------
players = sorted(df[player_col].dropna().unique())

selected_player = st.selectbox(
    "Select Player",
    players
)

# -----------------------------
# Prediction logic
# -----------------------------
if st.button("Predict Runs"):

    # Filter player rows
    player_rows = df[df[player_col] == selected_player]

    if len(player_rows) == 0:
        st.error("No data found for this player")
        st.stop()

    # Take numeric mean (same approach as baseline)
    player_features = player_rows.mean(numeric_only=True)

    # Convert to dataframe
    input_df = pd.DataFrame([player_features])

    # Align with training features
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    # Display
    st.success(f"Predicted Runs: {prediction:.2f}")

    # Show confidence context
    st.info("Prediction based on historical performance patterns")

# -----------------------------
# Debug expander (optional)
# -----------------------------
with st.expander("Debug Info"):
    st.write("Dataset shape:", df.shape)
    st.write("Player column used:", player_col)
    st.write("Model expects features:", len(feature_columns))
