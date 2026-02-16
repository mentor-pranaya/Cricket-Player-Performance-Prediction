import streamlit as st
import joblib
import pandas as pd

# Load models
batsman_model = joblib.load("xgb_batsman_model.pkl")
bowler_model = joblib.load("xgb_bowler_model.pkl")

batsman_features = joblib.load("batsman_features.pkl")
bowler_features = joblib.load("bowler_features.pkl")

st.title("🏏 Cricket Player Performance Prediction")

model_type = st.selectbox("Select Model Type", ["Batsman Runs", "Bowler Wickets"])

if model_type == "Batsman Runs":

    st.subheader("Predict Runs")

    input_data = {}
    for feature in batsman_features:
        input_data[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Runs"):
        input_df = pd.DataFrame([input_data])
        prediction = batsman_model.predict(input_df)
        st.success(f"Predicted Runs: {prediction[0]:.2f}")

else:

    st.subheader("Predict Wickets")

    input_data = {}
    for feature in bowler_features:
        input_data[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Wickets"):
        input_df = pd.DataFrame([input_data])
        prediction = bowler_model.predict(input_df)
        st.success(f"Predicted Wickets: {prediction[0]:.2f}")
