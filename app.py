# ==============================================================================
# FILE: app.py
# DESCRIPTION: IPL AI Predictor — Full Dashboard with Predictions, Form Charts,
#              Feature Importance (SHAP), and Career Analytics
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap

st.set_page_config(page_title="IPL AI Predictor", page_icon="🏏", layout="wide")

# ==============================================================================
# 1. SETUP & MAPPINGS
# ==============================================================================
VENUE_TO_CITY = {
    'M Chinnaswamy Stadium': 'Bangalore',
    'Wankhede Stadium': 'Mumbai',
    'Eden Gardens': 'Kolkata',
    'MA Chidambaram Stadium, Chepauk': 'Chennai',
    'Narendra Modi Stadium': 'Ahmedabad',
    'Arun Jaitley Stadium': 'Delhi',
    'Rajiv Gandhi International Stadium, Uppal': 'Hyderabad',
    'Sawai Mansingh Stadium': 'Jaipur',
    'Punjab Cricket Association IS Bindra Stadium': 'Mohali',
    'Lucknow Cricket Stadium': 'Lucknow'
}

TEAM_HOME_CITIES = {
    'Royal Challengers Bangalore': 'Bangalore', 'Royal Challengers Bengaluru': 'Bangalore',
    'Mumbai Indians': 'Mumbai',
    'Kolkata Knight Riders': 'Kolkata',
    'Chennai Super Kings': 'Chennai',
    'Gujarat Titans': 'Ahmedabad',
    'Delhi Capitals': 'Delhi', 'Delhi Daredevils': 'Delhi',
    'Sunrisers Hyderabad': 'Hyderabad', 'Deccan Chargers': 'Hyderabad',
    'Rajasthan Royals': 'Jaipur',
    'Punjab Kings': 'Mohali', 'Kings XI Punjab': 'Mohali',
    'Lucknow Super Giants': 'Lucknow'
}

CURRENT_TEAMS = [
    'Royal Challengers Bengaluru', 'Mumbai Indians', 'Kolkata Knight Riders',
    'Chennai Super Kings', 'Gujarat Titans', 'Delhi Capitals',
    'Sunrisers Hyderabad', 'Rajasthan Royals', 'Punjab Kings',
    'Lucknow Super Giants'
]

# ==============================================================================
# 2. LOAD ARTIFACTS
# ==============================================================================
@st.cache_resource
def load_data():
    bat_model = joblib.load('batsman_pipeline.pkl')
    bowl_model = joblib.load('bowler_pipeline.pkl')

    bat_base = pd.read_csv('prod_batsman_base.csv')
    bowl_base = pd.read_csv('prod_bowler_base.csv')

    bat_opp = pd.read_csv('prod_bat_vs_opp.csv')
    bowl_opp = pd.read_csv('prod_bowl_vs_opp.csv')
    bat_ven = pd.read_csv('prod_bat_at_venue.csv')
    bowl_ven = pd.read_csv('prod_bowl_at_venue.csv')

    # Full feature data for form charts and SHAP
    bat_features = pd.read_csv('batsman_features_final.csv')
    bowl_features = pd.read_csv('bowler_features_final.csv')

    return (bat_model, bowl_model, bat_base, bowl_base,
            bat_opp, bowl_opp, bat_ven, bowl_ven,
            bat_features, bowl_features)

try:
    (bat_model, bowl_model, bat_base, bowl_base,
     bat_opp, bowl_opp, bat_ven, bowl_ven,
     bat_features, bowl_features) = load_data()
except Exception as e:
    st.error(f"❌ Critical Error: {e}")
    st.stop()

# ==============================================================================
# 3. SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("🏏 IPL AI Predictor")
mode = st.sidebar.radio("Predict For:", ["Batsman", "Bowler"])

if mode == "Batsman":
    players = sorted(bat_base['player'].unique())
    idx = players.index('V Kohli') if 'V Kohli' in players else 0
    player = st.sidebar.selectbox("Player", players, index=idx)
    base_row = bat_base[bat_base['player'] == player].iloc[0]
else:
    players = sorted(bowl_base['player'].unique())
    idx = players.index('JJ Bumrah') if 'JJ Bumrah' in players else 0
    player = st.sidebar.selectbox("Player", players, index=idx)
    base_row = bowl_base[bowl_base['player'] == player].iloc[0]

current_team = base_row['team']
st.sidebar.info(f"Team: **{current_team}**")

opponents = sorted([t for t in CURRENT_TEAMS if t != current_team])
selected_opponent = st.sidebar.selectbox("Opponent", opponents)
selected_venue = st.sidebar.selectbox("Venue", sorted(VENUE_TO_CITY.keys()))
selected_innings = st.sidebar.radio("Innings", [1, 2])

# ==============================================================================
# 4. PREDICTION LOGIC
# ==============================================================================
if st.sidebar.button("🚀 Run Prediction", type="primary"):

    # Start with base row — pd.DataFrame([base_row]) preserves dtypes
    input_data = pd.DataFrame([base_row])

    # --- DYNAMIC INJECTION ---
    input_data['opponent'] = selected_opponent
    input_data['venue'] = selected_venue
    input_data['innings'] = selected_innings

    # Derive batting_first / bowling_first from selected innings
    if mode == "Batsman":
        input_data['batting_first'] = 1 if selected_innings == 1 else 0
    else:
        input_data['bowling_first'] = 1 if selected_innings == 1 else 0

    # Lookup Stats
    if mode == "Batsman":
        opp_stats = bat_opp[(bat_opp['player'] == player) & (bat_opp['opponent'] == selected_opponent)]
        ven_stats = bat_ven[(bat_ven['player'] == player) & (bat_ven['venue'] == selected_venue)]

        if not opp_stats.empty:
            input_data['vs_opponent_runs_avg'] = opp_stats['vs_opponent_runs_avg'].values[0]
            input_data['vs_opponent_strike_rate_avg'] = opp_stats['vs_opponent_strike_rate_avg'].values[0]
        else:
            input_data['vs_opponent_runs_avg'] = base_row['career_avg_runs']
            input_data['vs_opponent_strike_rate_avg'] = base_row['career_strike_rate']

        if not ven_stats.empty:
            input_data['venue_runs_avg'] = ven_stats['venue_runs_avg'].values[0]
            input_data['venue_strike_rate_avg'] = ven_stats['venue_strike_rate_avg'].values[0]
        else:
            input_data['venue_runs_avg'] = base_row['career_avg_runs']
            input_data['venue_strike_rate_avg'] = base_row['career_strike_rate']

    else:  # Bowler Logic
        opp_stats = bowl_opp[(bowl_opp['player'] == player) & (bowl_opp['opponent'] == selected_opponent)]
        ven_stats = bowl_ven[(bowl_ven['player'] == player) & (bowl_ven['venue'] == selected_venue)]

        if not opp_stats.empty:
            input_data['vs_opponent_wickets_avg'] = opp_stats['vs_opponent_wickets_avg'].values[0]
            input_data['vs_opponent_economy_avg'] = opp_stats['vs_opponent_economy_avg'].values[0]
        else:
            input_data['vs_opponent_wickets_avg'] = base_row['career_avg_wickets']
            input_data['vs_opponent_economy_avg'] = base_row['career_economy']

        if not ven_stats.empty:
            input_data['venue_wickets_avg'] = ven_stats['venue_wickets_avg'].values[0]
            input_data['venue_economy_avg'] = ven_stats['venue_economy_avg'].values[0]
        else:
            input_data['venue_wickets_avg'] = base_row['career_avg_wickets']
            input_data['venue_economy_avg'] = base_row['career_economy']

    # Contextual Logic
    city = VENUE_TO_CITY.get(selected_venue, 'Other')
    input_data['city'] = city
    home_city = TEAM_HOME_CITIES.get(current_team, 'None')
    input_data['is_home_match'] = 1 if city == home_city else 0

    # Sanitize Inputs
    cols_to_drop = [
        'runs', 'wickets', 'match_id', 'player', 'date', 'season',
        'target_runs', 'target_wickets', 'target_balls_faced', 'target_strike_rate',
        'target_runs_conceded', 'target_economy', 'balls_faced', 'strike_rate',
        'boundaries', 'got_out', 'runs_conceded', 'economy', 'balls_bowled', 'overs_bowled',
        'next_opponent', 'next_venue'
    ]

    existing_drop = [c for c in cols_to_drop if c in input_data.columns]
    input_data_clean = input_data.drop(columns=existing_drop)
    input_data_clean = input_data_clean.fillna(0)

    # =========================================================
    # 5. DISPLAY RESULTS
    # =========================================================
    st.divider()
    st.header(f"🎯 Prediction for {player}")

    try:
        if mode == "Batsman":
            pred = bat_model.predict(input_data_clean)[0]
            pred = max(0, pred)

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Runs", f"{int(round(pred))}")
            col2.metric("Vs Opponent Avg", f"{input_data['vs_opponent_runs_avg'].values[0]:.1f}")
            col3.metric("At Venue Avg", f"{input_data['venue_runs_avg'].values[0]:.1f}")

        else:  # Bowler
            pred = bowl_model.predict(input_data_clean)[0]
            pred = max(0, pred)

            if pred < 0.8:
                rounded_wickets = 0
            elif pred < 1.5:
                rounded_wickets = 1
            else:
                rounded_wickets = int(round(pred))

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Wickets", f"{pred:.2f} ({rounded_wickets})")
            col2.metric("Vs Opponent Avg", f"{input_data['vs_opponent_wickets_avg'].values[0]:.2f}")
            col3.metric("At Venue Avg", f"{input_data['venue_wickets_avg'].values[0]:.2f}")
            st.caption(f"Raw Prediction: {pred:.3f} | Bracket value is the likely integer wickets.")

        # =========================================================
        # 6. FORM TREND CHART
        # =========================================================
        st.divider()
        st.subheader("📈 Recent Form Trend")

        if mode == "Batsman":
            full_data = bat_features[bat_features['player'] == player].sort_values('date').tail(20)
            if not full_data.empty:
                fig_form = go.Figure()
                fig_form.add_trace(go.Bar(
                    x=list(range(1, len(full_data) + 1)),
                    y=full_data['runs'],
                    name='Runs Scored',
                    marker_color='rgba(55, 128, 191, 0.7)',
                    text=full_data['runs'].astype(int),
                    textposition='outside'
                ))
                fig_form.add_trace(go.Scatter(
                    x=list(range(1, len(full_data) + 1)),
                    y=full_data['runs_last_5'],
                    name='5-Match Avg',
                    mode='lines+markers',
                    line=dict(color='#FF6347', width=3),
                    marker=dict(size=6)
                ))
                fig_form.add_hline(
                    y=base_row['career_avg_runs'],
                    line_dash="dash", line_color="green",
                    annotation_text=f"Career Avg: {base_row['career_avg_runs']:.1f}"
                )
                fig_form.update_layout(
                    xaxis_title="Match #", yaxis_title="Runs",
                    template="plotly_white", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig_form, use_container_width=True)
            else:
                st.info("No recent match data available for this player.")
        else:
            full_data = bowl_features[bowl_features['player'] == player].sort_values('date').tail(20)
            if not full_data.empty:
                fig_form = go.Figure()
                fig_form.add_trace(go.Bar(
                    x=list(range(1, len(full_data) + 1)),
                    y=full_data['wickets'],
                    name='Wickets Taken',
                    marker_color='rgba(219, 64, 82, 0.7)',
                    text=full_data['wickets'].astype(int),
                    textposition='outside'
                ))
                fig_form.add_trace(go.Scatter(
                    x=list(range(1, len(full_data) + 1)),
                    y=full_data['wickets_last_5'],
                    name='5-Match Avg',
                    mode='lines+markers',
                    line=dict(color='#4169E1', width=3),
                    marker=dict(size=6)
                ))
                fig_form.add_hline(
                    y=base_row['career_avg_wickets'],
                    line_dash="dash", line_color="green",
                    annotation_text=f"Career Avg: {base_row['career_avg_wickets']:.2f}"
                )
                fig_form.update_layout(
                    xaxis_title="Match #", yaxis_title="Wickets",
                    template="plotly_white", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig_form, use_container_width=True)
            else:
                st.info("No recent match data available for this player.")

        # =========================================================
        # 7. FEATURE IMPORTANCE (SHAP)
        # =========================================================
        st.divider()
        st.subheader("🔍 Feature Importance (SHAP)")

        try:
            if mode == "Batsman":
                model_pipeline = bat_model
            else:
                model_pipeline = bowl_model

            # Extract the fitted RF model and preprocessor from the pipeline
            fitted_preprocessor = model_pipeline.named_steps['preprocessor']
            fitted_rf = model_pipeline.named_steps['regressor']

            # Get feature names after one-hot encoding
            feature_names = fitted_preprocessor.get_feature_names_out()
            # Clean up names (remove 'num__' and 'cat__' prefixes)
            feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]

            # Transform the input and compute SHAP values
            input_transformed = fitted_preprocessor.transform(input_data_clean)
            explainer = shap.TreeExplainer(fitted_rf)
            shap_values = explainer.shap_values(input_transformed)

            # Get the SHAP values for this single prediction
            sv = shap_values[0]  # single row

            # Create a DataFrame of feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': sv
            })
            importance_df['Abs SHAP'] = importance_df['SHAP Value'].abs()
            importance_df = importance_df.sort_values('Abs SHAP', ascending=True).tail(15)

            # Plot
            colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in importance_df['SHAP Value']]
            fig_shap = go.Figure(go.Bar(
                x=importance_df['SHAP Value'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color=colors
            ))
            fig_shap.update_layout(
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("🟢 Green = pushes prediction higher | 🔴 Red = pushes prediction lower")

        except Exception as shap_err:
            st.warning(f"Could not compute SHAP: {shap_err}")

        # =========================================================
        # 8. CAREER STATS SUMMARY
        # =========================================================
        st.divider()
        st.subheader("📊 Player Profile & Stats")

        if mode == "Batsman":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Career Matches", f"{int(base_row['career_matches'])}")
            c2.metric("Career Avg Runs", f"{base_row['career_avg_runs']:.1f}")
            c3.metric("Career Strike Rate", f"{base_row['career_strike_rate']:.1f}")
            c4.metric("Experience", base_row['experience_level'].title())

            # Venue & Opponent breakdown
            vcol, ocol = st.columns(2)
            with vcol:
                st.markdown("**Venue History**")
                p_venues = bat_ven[bat_ven['player'] == player].sort_values('venue_runs_avg', ascending=False).head(8)
                if not p_venues.empty:
                    fig_v = px.bar(p_venues, x='venue', y='venue_runs_avg',
                                  color='venue_strike_rate_avg',
                                  color_continuous_scale='RdYlGn',
                                  labels={'venue_runs_avg': 'Avg Runs', 'venue': 'Venue',
                                          'venue_strike_rate_avg': 'SR'})
                    fig_v.update_layout(template='plotly_white', height=350,
                                       xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig_v, use_container_width=True)
            with ocol:
                st.markdown("**Vs Opponent History**")
                p_opps = bat_opp[bat_opp['player'] == player].sort_values('vs_opponent_runs_avg', ascending=False).head(8)
                if not p_opps.empty:
                    fig_o = px.bar(p_opps, x='opponent', y='vs_opponent_runs_avg',
                                  color='vs_opponent_strike_rate_avg',
                                  color_continuous_scale='RdYlGn',
                                  labels={'vs_opponent_runs_avg': 'Avg Runs', 'opponent': 'Opponent',
                                          'vs_opponent_strike_rate_avg': 'SR'})
                    fig_o.update_layout(template='plotly_white', height=350,
                                       xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig_o, use_container_width=True)
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Career Matches", f"{int(base_row['career_matches'])}")
            c2.metric("Career Avg Wickets", f"{base_row['career_avg_wickets']:.2f}")
            c3.metric("Career Economy", f"{base_row['career_economy']:.2f}")
            c4.metric("Experience", base_row['experience_level'].title())

            vcol, ocol = st.columns(2)
            with vcol:
                st.markdown("**Venue History**")
                p_venues = bowl_ven[bowl_ven['player'] == player].sort_values('venue_wickets_avg', ascending=False).head(8)
                if not p_venues.empty:
                    fig_v = px.bar(p_venues, x='venue', y='venue_wickets_avg',
                                  color='venue_economy_avg',
                                  color_continuous_scale='RdYlGn_r',
                                  labels={'venue_wickets_avg': 'Avg Wickets', 'venue': 'Venue',
                                          'venue_economy_avg': 'Economy'})
                    fig_v.update_layout(template='plotly_white', height=350,
                                       xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig_v, use_container_width=True)
            with ocol:
                st.markdown("**Vs Opponent History**")
                p_opps = bowl_opp[bowl_opp['player'] == player].sort_values('vs_opponent_wickets_avg', ascending=False).head(8)
                if not p_opps.empty:
                    fig_o = px.bar(p_opps, x='opponent', y='vs_opponent_wickets_avg',
                                  color='vs_opponent_economy_avg',
                                  color_continuous_scale='RdYlGn_r',
                                  labels={'vs_opponent_wickets_avg': 'Avg Wickets', 'opponent': 'Opponent',
                                          'vs_opponent_economy_avg': 'Economy'})
                    fig_o.update_layout(template='plotly_white', height=350,
                                       xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig_o, use_container_width=True)

        # Debug Expander
        with st.expander("🔍 See Model Inputs (Debug)"):
            st.write("The model is seeing these values:")
            st.dataframe(input_data_clean)

    except Exception as e:
        st.error(f"Prediction Failed: {e}")
        with st.expander("Error Details"):
            st.write("Input Columns:", input_data_clean.columns.tolist())

else:
    st.info("👈 Set match parameters and click Run")