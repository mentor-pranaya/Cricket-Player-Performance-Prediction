import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

st.set_page_config(page_title="Cricket Performance", layout="wide")

st.markdown("""
<style>
.main-title{text-align:center;font-size:28px;font-weight:700;margin-bottom:10px;}
.block{background:#0f223a;padding:18px;border-radius:12px;margin-bottom:12px;}
.metric-card{background:#132a46;padding:20px;border-radius:12px;text-align:center;}
.metric-value{font-size:42px;font-weight:800;}
.metric-label{color:#9bb3d3;}
.conf{font-size:13px;color:#9bb3d3;margin-top:4px;}
.section-title{font-size:18px;font-weight:700;margin-bottom:6px;color:#6EC1FF;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">CRICKET PLAYER PERFORMANCE PREDICTION</div>', unsafe_allow_html=True)

# DATA
batsman_df = pd.read_csv("DATA1/final_dataset.csv")
bowler_df = pd.read_csv("DATA1/bowler_dataset.csv")

runs_model = joblib.load("models1/xgb_model.joblib")
pipeline = joblib.load("models1/feature_pipeline.pkl")
wickets_model = joblib.load("models1/wicket_model.joblib")

RUN_FEATURES = [
    "career_avg_runs","recent_form","recent_form_5",
    "venue_avg_runs","opponent_avg","matches_played",
    "run_consistency","fifty_rate"
]

WICKET_FEATURES = [
    "career_wicket_avg","recent_wickets","opponent_wicket_avg"
]

players = sorted(list(set(batsman_df["batsman"]).union(set(bowler_df["bowler"]))))

left, right = st.columns([1,3])

# INPUT PANEL
with left:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input Parameters</div>', unsafe_allow_html=True)

    player = st.selectbox("Player", players)
    opponent = st.selectbox("Opponent", sorted(batsman_df["opponent"].unique()))
    venue = st.selectbox("Venue", sorted(batsman_df["venue"].unique()))
    predict = st.button("Predict Performance")
    st.markdown('</div>', unsafe_allow_html=True)

# DASHBOARD
with right:
    if predict:

        pred_runs, pred_wickets = None, None
        runs_conf, wickets_conf = "-", "-"

        # RUNS
        if player in batsman_df["batsman"].values:
            row = batsman_df[
                (batsman_df["batsman"]==player) &
                (batsman_df["opponent"]==opponent) &
                (batsman_df["venue"]==venue)
            ]
            if row.empty:
                row = batsman_df[batsman_df["batsman"]==player].tail(1)

            X_df = row[RUN_FEATURES]
            X = pipeline.transform(X_df)
            pred_runs = int(runs_model.predict(X)[0])

            # confidence (UI only)
            if pred_runs > 40: runs_conf = "High"
            elif pred_runs > 25: runs_conf = "Medium"
            else: runs_conf = "Low"

        # WICKETS
        if player in bowler_df["bowler"].values:
            brow = bowler_df[
                (bowler_df["bowler"]==player) &
                (bowler_df["opponent"]==opponent)
            ]
            if brow.empty:
                brow = bowler_df[bowler_df["bowler"]==player].tail(1)

            pred_wickets = int(wickets_model.predict(brow[WICKET_FEATURES])[0])

            if pred_wickets >= 3: wickets_conf = "High"
            elif pred_wickets == 2: wickets_conf = "Medium"
            else: wickets_conf = "Low"

        # KPI CARDS
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Predicted Runs</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{pred_runs if pred_runs is not None else "-"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="conf">Confidence: {runs_conf}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Predicted Wickets</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{pred_wickets if pred_wickets is not None else "-"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="conf">Confidence: {wickets_conf}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # FORM + SHAP
        g1, g2 = st.columns(2)

        if pred_runs is not None:
            with g1:
                st.markdown('<div class="block">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Player Form (Last 5 Matches)</div>', unsafe_allow_html=True)
                last5 = batsman_df[batsman_df["batsman"]==player]["runs"].tail(5)
                fig, ax = plt.subplots(figsize=(5,2.5))
                ax.plot(range(1,len(last5)+1), last5.values, marker="o")
                ax.set_title("Runs Trend", color="#6EC1FF")
                ax.set_xlabel("Match Number")
                ax.set_ylabel("Runs")
                ax.grid(alpha=0.3)
                st.pyplot(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if pred_runs is not None:
            with g2:
                st.markdown('<div class="block">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">SHAP Feature Importance</div>', unsafe_allow_html=True)
                explainer = shap.Explainer(runs_model)
                shap_values = explainer(X)
                shap_values.feature_names = RUN_FEATURES
                fig_shap, ax = plt.subplots(figsize=(5,2.5))
                shap.plots.bar(shap_values, show=False)
                plt.title("Feature Impact", color="#6EC1FF")
                st.pyplot(fig_shap, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ANALYTICAL REPORT
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analytical Report</div>', unsafe_allow_html=True)

        sample = batsman_df[batsman_df["batsman"]==player].tail(10)
        Xs = pipeline.transform(sample[RUN_FEATURES])
        preds = runs_model.predict(Xs)

        report = pd.DataFrame({
            "Actual Runs": sample["runs"].values,
            "Predicted Runs": preds
        })

        st.dataframe(report, use_container_width=True)

        r1, r2 = st.columns(2)

        with r1:
            fig1, ax1 = plt.subplots(figsize=(5,2.5))
            ax1.scatter(report["Actual Runs"], report["Predicted Runs"])
            ax1.set_title("Actual vs Predicted", color="#6EC1FF")
            ax1.set_xlabel("Actual Runs")
            ax1.set_ylabel("Predicted Runs")
            ax1.grid(alpha=0.3)
            st.pyplot(fig1, use_container_width=True)

        with r2:
            residuals = report["Actual Runs"] - report["Predicted Runs"]
            fig2, ax2 = plt.subplots(figsize=(5,2.5))
            sns.histplot(residuals, kde=True, ax=ax2)
            ax2.set_title("Residual Distribution", color="#6EC1FF")
            ax2.set_xlabel("Prediction Error")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
