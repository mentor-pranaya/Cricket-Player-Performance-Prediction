# =========================================================
# IPL PLAYER PERFORMANCE PREDICTION – FINAL DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# GLOBAL GRAPH STYLE (WHITE BACKGROUND, NO GRID)
# ---------------------------------------------------------
plt.style.use("default")
sns.set_style("white")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cricket Player Performance Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)
## =========================================================
# RIGHT-SIDE BACKGROUND IMAGE (DARKER & FULL HEIGHT)
# =========================================================
import base64

def add_main_background():
    image_path = r"C:\Users\sinchana k\Downloads\cric jpg.jpg"

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    to right,
                    rgba(255,255,255,1) 0%,
                    rgba(255,255,255,1) 50%,
                    rgba(255,255,255,0.7) 70%,
                    rgba(255,255,255,0.7) 100%
                ),
                url("data:image/jpg;base64,{encoded}");
            background-repeat: no-repeat;
            background-position: right center;
            background-size: 45% 100%;
        }}

        /* Keep text dark and readable */
        h1, h2, h3, h4, p, label {{
            color: #111 !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_main_background()





# =========================================================
# LOAD DATA & MODEL
# =========================================================
df = pd.read_csv("data/cleaned/dataset.csv")

# ✅ FIX: convert date column properly (prevents .dt error)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

pipeline = joblib.load("models/final_pipeline.joblib")

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<h1 style='text-align:center; color:#1f77b4; font-size:42px;'>
🏏 IPL Player Performance Prediction
</h1>
<p style='text-align:center; color:#555; font-size:18px;'>
Machine Learning Dashboard for IPL Player Analysis
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("🎯 Match Inputs")

batsman = st.sidebar.selectbox("Select Batsman", sorted(df["batsman"].unique()))
venue = st.sidebar.selectbox("Select Venue", sorted(df["venue"].unique()))
batting_team = st.sidebar.selectbox("Batting Team", sorted(df["batting_team"].unique()))
bowling_team = st.sidebar.selectbox("Bowling Team", sorted(df["bowling_team"].unique()))

predict_btn = st.sidebar.button("🚀 Predict Performance")

# =========================================================
# PLAYER DATA
# =========================================================
player_df = df[df["batsman"] == batsman].sort_values("date")
latest = player_df.iloc[-1]

# =========================================================
# MODEL INPUT (AUTO-FEATURES)


# =========================================================
input_df = pd.DataFrame([{
    "runs_scored": latest["runs_scored"],
    "balls_faced": latest["balls_faced"],
    "strike_rate": latest["strike_rate"],
    "avg_runs_last_3": latest["avg_runs_last_3"],
    "avg_runs_last_5": latest["avg_runs_last_5"],
    "venue_avg_runs": latest["venue_avg_runs"],
    "career_runs": latest["career_runs"],
    "career_avg": latest["career_avg"],
    "venue": venue,
    "batting_team": batting_team,
    "bowling_team": bowling_team,
    "batsman": batsman
}])

# =========================================================
# MODEL METRICS
st.markdown("## 🎯 Match Prediction Summary")
# =========================================================
X_all = df.drop(columns=["next_match_runs"])
y_true = df["next_match_runs"]
y_pred = pipeline.predict(X_all)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Players", df["batsman"].nunique())
m2.metric("Total Matches", len(df))
m3.metric("MAE", f"{np.mean(np.abs(y_true - y_pred)):.2f}")
m4.metric("RMSE", f"{np.sqrt(np.mean((y_true - y_pred) ** 2)):.2f}")

st.markdown("---")

# =========================================================
# PREDICTION
st.markdown("## 📈 Player Performance Analysis")

# =========================================================
if predict_btn:
    prediction = pipeline.predict(input_df)[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏏 Predicted Runs", round(prediction, 1))
    c2.metric("Career Avg", round(latest["career_avg"], 1))
    c3.metric("Last 5 Avg", round(latest["avg_runs_last_5"], 1))
    c4.metric("Strike Rate", round(latest["strike_rate"], 1))

st.markdown("---")

# ==================================

st.markdown(
    "<p style='color:#555;'>Visual analysis of recent form, consistency, and batting behavior</p>",
    unsafe_allow_html=True
)

# =========================================================
# GRAPH 1 – LAST 5 MATCH RUNS
# =========================================================
st.subheader("📈 Last 5 Matches – Runs")

last5 = player_df.tail(5)
fig1, ax1 = plt.subplots(figsize=(8, 4)
)
ax1.plot(last5["date"], last5["runs_scored"], marker="o")
ax1.set_ylabel("Runs")
ax1.grid(False)
plt.xticks(rotation=45)
st.pyplot(fig1)

# =========================================================
# GRAPH 2 – LAST 10 MATCH RUNS
# =========================================================
st.subheader("📈 Last 10 Matches – Runs")

last10 = player_df.tail(10)
fig2, ax2 =plt.subplots(figsize=(8, 4)
)
ax2.plot(last10["date"], last10["runs_scored"], marker="o")
ax2.set_ylabel("Runs")
ax2.grid(False)
plt.xticks(rotation=45)
st.pyplot(fig2)

# =========================================================
# GRAPH 3 – RUN DISTRIBUTION
# =========================================================
st.subheader("📊 Run Distribution")

fig3, ax3 =plt.subplots(figsize=(8, 4)
)
sns.histplot(player_df["runs_scored"], bins=20, kde=True, ax=ax3)
ax3.grid(False)
st.pyplot(fig3)

# =========================================================
# GRAPH 4 – ACTUAL VS PREDICTED (REGRESSION SCATTER)
# =========================================================
st.subheader("📉 Actual vs Predicted Runs")

fig4, ax4 =plt.subplots(figsize=(8, 4)
)

# Scatter plot
ax4.scatter(
    y_true,
    y_pred,
    alpha=0.35,
    s=20,
    color="steelblue"
)


# Best-fit regression line
m, b = np.polyfit(y_true, y_pred, 1)
ax4.plot(
    y_true,
    m * y_true + b,
    color="red",
    linewidth=2,
    label="Best Fit Line"
)

ax4.set_xlabel("Actual Runs")
ax4.set_ylabel("Predicted Runs")
ax4.legend()
ax4.grid(False)

plt.tight_layout()
st.pyplot(fig4)


# =========================================================
# GRAPH 5 – RESIDUALS
# =========================================================
st.subheader("📉 Residuals vs Predicted")

residuals = y_true - y_pred
fig5, ax5 = plt.subplots(figsize=(8, 4)
)
sns.scatterplot(x=y_pred, y=residuals, ax=ax5, alpha=0.4)
ax5.axhline(0, color="red", linestyle="--")
ax5.set_xlabel("Predicted Runs")
ax5.set_ylabel("Residuals")
ax5.grid(False)
st.pyplot(fig5)

# =========================================================
# GRAPH 6 – BALLS FACED VS RUNS
# =========================================================
st.subheader("⚾ Balls Faced vs Runs")

fig6, ax6 = plt.subplots(figsize=(8, 4)
)
sns.scatterplot(x=player_df["balls_faced"], y=player_df["runs_scored"], ax=ax6)
ax6.set_xlabel("Balls Faced")
ax6.set_ylabel("Runs")
ax6.grid(False)
st.pyplot(fig6)

# =========================================================
# GRAPH 7 – STRIKE RATE TREND (CLEAN X-AXIS)
# =========================================================
st.subheader("⚡ Strike Rate Trend")

fig7, ax7 = plt.subplots(figsize=(8, 4)
)

# Use match order instead of date (cleanest visualization)
match_numbers = range(1, len(player_df) + 1)

ax7.plot(
    match_numbers,
    player_df["strike_rate"],
    marker="o",
    linewidth=2,
    color="orange"
)

ax7.set_xlabel("Match Number")
ax7.set_ylabel("Strike Rate")
ax7.grid(False)

plt.tight_layout()
st.pyplot(fig7)



# =========================================================
# GRAPH 8 – ROLLING AVERAGE FORM
# =========================================================
st.subheader("📉 Rolling Average Runs")

fig8, ax8 = plt.subplots(figsize=(8, 4)
)
ax8.plot(player_df["date"], player_df["avg_runs_last_3"], label="Last 3 Matches")
ax8.plot(player_df["date"], player_df["avg_runs_last_5"], label="Last 5 Matches")
ax8.legend()
ax8.set_ylabel("Average Runs")
ax8.grid(False)
plt.xticks(rotation=45)
st.pyplot(fig8)

# =========================================================
# GRAPH 9 – CONSISTENCY TREND (ROLLING STANDARD DEVIATION)
# =========================================================
st.subheader("📉 Consistency of Performance (Stability Over Matches)")

# Rolling standard deviation over last 5 matches
player_df["rolling_std_5"] = (
    player_df["runs_scored"]
    .rolling(window=5, min_periods=2)
    .std()
)

fig9, ax9 =plt.subplots(figsize=(8,4)
)

match_numbers = range(1, len(player_df) + 1)

ax9.plot(
    match_numbers,
    player_df["rolling_std_5"],
    marker="o",
    linewidth=2,
    color="purple"
)

ax9.set_xlabel("Match Number")
ax9.set_ylabel("Run Variability (Std Dev)")
ax9.set_title("Lower Value = More Consistent Performance")
ax9.grid(False)

plt.tight_layout()
st.pyplot(fig9)


# =========================================================
# SAMPLE PREDICTIONS TABLE
# =========================================================
st.subheader("📋 Model Validation – Sample Predictions")
st.markdown(
    "<p style='color:#666;'>Random historical samples used to evaluate model accuracy (not related to selected player)</p>",
    unsafe_allow_html=True
)

sample = df.sample(8, random_state=1)
sample_X = sample.drop(columns=["next_match_runs"])
sample["Predicted"] = pipeline.predict(sample_X).round(1)
sample["Actual"] = sample["next_match_runs"]
sample["Error"] = sample["Actual"] - sample["Predicted"]

st.dataframe(
    sample[["batsman", "venue", "Actual", "Predicted", "Error"]],
    width="stretch"
)


# =========================================================
# PLAYER INSIGHTS & MODEL INTERPRETATION
# =========================================================
st.markdown("---")
st.subheader("🧠 Player Performance Insights")

st.markdown(f"""
**Player Analysis Summary**

- The predicted runs for **{batsman}** are based on the player’s **recent form**, **career consistency**,  
  and **venue-specific performance patterns** learned by the machine learning model.

- Over the **last 5 matches**, the player has an average of **{round(latest['avg_runs_last_5'], 1)} runs**,  
  indicating the current batting form.

- The **career average of {round(latest['career_avg'], 1)} runs** suggests the long-term reliability  
  of the player across multiple seasons.

- The model also considers how the player historically performs at the selected **venue ({venue})**,  
  which plays a significant role in adjusting the prediction.

- Variations in prediction error (seen in the table above) occur due to match situations,  
  opposition bowling strength, and inherent uncertainty in sports performance.
""")

st.info(
    "📌 **Note:** Predictions are probabilistic estimates, not exact outcomes. "
    "The model aims to capture trends and patterns rather than guarantee match results."
)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("ML-Based IPL Player Performance Prediction | Streamlit Dashboard")