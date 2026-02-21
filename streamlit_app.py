import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("batsman_xgb_model.pkl", "rb"))

# Load feature columns
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# Load dataset
df = pd.read_csv("dataset_batting_features_v2.csv")

st.title("Cricket Player Performance Predictor")

player = st.selectbox("Select Player", sorted(df["batter"].unique()))
venue = st.selectbox("Select Venue", sorted(df["venue"].unique()))
opponent = st.selectbox("Select Opponent", sorted(df["bowling_team"].unique()))

# Prepare input
player_data = df[df["batter"] == player].mean(numeric_only=True)

input_df = pd.DataFrame([player_data])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Predict
if st.button("Predict Runs"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Runs: {prediction:.2f}")
