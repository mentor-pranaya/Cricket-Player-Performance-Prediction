import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("IPL AI Analytics Dashboard")

bat_model = pickle.load(open("models/bat_model.pkl","rb"))

data = pd.read_csv("data/final_batsman.csv")

st.title("IPL Player Prediction")

st.header("Compare Players")

p1 = st.selectbox("Player 1",data["batsman"].unique())
p2 = st.selectbox("Player 2",data["batsman"].unique())

st.write(data[data["batsman"]==p1]["career_avg_runs"].mean())
st.write(data[data["batsman"]==p2]["career_avg_runs"].mean())

player = st.selectbox("Select Player",data["batsman"].unique())

latest = data[data["batsman"]==player].iloc[-1]

features = latest[["career_avg_runs","prev_10_avg_runs","strike_rate","fours","sixes"]]

prediction = bat_model.predict([features])[0]

log = pd.DataFrame([[player,prediction]],columns=["player","prediction"])
log.to_csv("prediction_log.csv",mode="a",header=False,index=False)

st.metric("Predicted Runs", round(prediction,2))

import shap

explainer = shap.TreeExplainer(bat_model)
input_df = pd.DataFrame([features.values], columns=features.index)

shap_values = explainer.shap_values(input_df)

st.subheader("Why this prediction?")
st.write(pd.DataFrame(shap_values, columns=input_df.columns))

fig, ax = plt.subplots()
shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
st.pyplot(fig)

recent = data[data["batsman"]==player].tail(10)
st.line_chart(recent["runs_scored"])

