import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="IPL Runs Predictor", layout="wide")

@st.cache_resource
def load_model():
    try:
        return joblib.load('xgb_model.pkl')
    except:
        st.error("Could not load xgb_model.pkl — check file exists")
        st.stop()

@st.cache_resource
def load_encoder():
    try:
        return joblib.load('venue_encoder.pkl')
    except:
        st.warning("venue_encoder.pkl not found → using fallback encoding")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('processed_player_data.csv')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except:
        st.error("processed_player_data.csv not found")
        st.stop()

model = load_model()
encoder = load_encoder()
data = load_data()

st.title("IPL Batsman Runs Predictor")
st.markdown("Choose a batsman, opponent, venue and season to predict runs.")

col1, col2, col3 = st.columns(3)

with col1:
    batsman = st.selectbox(
        "Batsman",
        sorted(data['batsman'].dropna().unique())
    )

with col2:
    opponent = st.selectbox(
        "Opponent (Bowling Team)",
        sorted(data['bowling_team'].dropna().unique())
    )

with col3:
    venue = st.selectbox(
        "Venue",
        sorted(data['venue'].dropna().unique())
    )

season_min = int(data['season_year'].min())
season_max = int(data['season_year'].max())

if season_min == season_max:
    season = st.number_input(
        "Season Year (only one available)",
        value=season_min,
        disabled=True
    )
    st.info(f"Using season {season_min} — only one season in data")
else:
    season = st.slider(
        "Target Season Year",
        min_value=season_min,
        max_value=season_max,
        value=season_max,
        step=1
    )

home = st.checkbox("Batting at Home?", value=True)

if st.button("Predict", type="primary", use_container_width=True):
    hist = data[
        (data['batsman'] == batsman) &
        (data['season_year'] < season)
    ]

    if len(hist) == 0:
        st.error(f"No past data for {batsman} before season {season}")
    else:
        batsman_avg = hist['runs_scored'].mean()
        hist_sr = hist.get('strike_rate', hist['runs_scored'] / hist['balls_faced'] * 100).mean()
        balls_avg = hist['balls_faced'].mean()

        vs_opp = hist[hist['bowling_team'] == opponent]
        avg_vs_opp = vs_opp['runs_scored'].mean() if not vs_opp.empty else batsman_avg

        if encoder is not None:
            try:
                venue_enc = int(encoder.transform([venue])[0])
            except:
                venue_enc = 0
        else:
            venue_enc = 0

        opp_past = data[
            (data['bowling_team'] == opponent) &
            (data['season_year'] < season)
        ]
        opp_strength = opp_past.groupby(['matchid', 'inning'])['runs_scored'].sum().mean()
        opp_strength = opp_strength if not np.isnan(opp_strength) else 155

        home_adv = 1 if home else 0

        input_row = pd.DataFrame({
            'batsman_avg': [batsman_avg],
            'historical_sr': [hist_sr],
            'balls_faced_avg': [balls_avg],
            'avg_vs_opp': [avg_vs_opp],
            'venue_encoded': [venue_enc],
            'opp_strength': [opp_strength],
            'home_advantage': [home_adv],
            'season_year': [season]
        })

        prediction = model.predict(input_row)[0]

        st.success(f"**Predicted Runs: {prediction:.1f}**")
        st.markdown(f"**{batsman}** vs **{opponent}** at **{venue}** in **{season}** (Home: {'Yes' if home else 'No'})")

        st.divider()

        with st.expander("Input Features Used"):
            st.dataframe(input_row.T.rename(columns={0: "Value"}))

        if not vs_opp.empty:
            st.subheader(f"Historical runs vs {opponent}")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(vs_opp['runs_scored'], kde=True, color='teal', ax=ax)
            ax.axvline(prediction, color='red', linestyle='--', label=f'Predicted: {prediction:.1f}')
            ax.set_title(f"{batsman} runs distribution vs {opponent}")
            ax.legend()
            st.pyplot(fig)
