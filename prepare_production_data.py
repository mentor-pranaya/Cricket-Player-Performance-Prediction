# ==============================================================================
# FILE: prepare_production_data.py
# DESCRIPTION: Generates Lookup Tables for Dynamic Dashboarding
# ==============================================================================
import pandas as pd
import numpy as np

print("⚙️ Generating Production Data Lookups...")

# Load full training data
batsman_df = pd.read_csv('batsman_features_final.csv')
bowler_df = pd.read_csv('bowler_features_final.csv')

# ---------------------------------------------------------
# 1. BASE PLAYER STATE (Most Recent Form)
# ---------------------------------------------------------
# We use the last row for rolling/career features (already smoothed),
# but OVERWRITE volatile single-match features with averages over
# the last 5 matches so the model input isn't dominated by one-match noise.

SMOOTHING_WINDOW = 5

# --- Batsman ---
batsman_sorted = batsman_df.sort_values('date')
batsman_base = batsman_sorted.groupby('player').tail(1).copy()

bat_volatile = ['form_indicator', 'performance_trend', 'consistency_score',
                'boundary_dependency', 'momentum_score']
bat_recent = batsman_sorted.groupby('player').tail(SMOOTHING_WINDOW)
bat_smooth = bat_recent.groupby('player')[bat_volatile].mean()
for col in bat_volatile:
    batsman_base = batsman_base.set_index('player')
    batsman_base[col] = bat_smooth[col]
    batsman_base = batsman_base.reset_index()

# --- Bowler ---
bowler_sorted = bowler_df.sort_values('date')
bowler_base = bowler_sorted.groupby('player').tail(1).copy()

bowl_volatile = ['form_indicator', 'performance_trend', 'consistency_score',
                 'wicket_taking_rate', 'momentum_score']
bowl_recent = bowler_sorted.groupby('player').tail(SMOOTHING_WINDOW)
bowl_smooth = bowl_recent.groupby('player')[bowl_volatile].mean()
for col in bowl_volatile:
    bowler_base = bowler_base.set_index('player')
    bowler_base[col] = bowl_smooth[col]
    bowler_base = bowler_base.reset_index()

# Save Base Stats
batsman_base.to_csv('prod_batsman_base.csv', index=False)
bowler_base.to_csv('prod_bowler_base.csv', index=False)
print("✅ Saved Base Stats (Smoothed over last 5 matches)")

# ---------------------------------------------------------
# 2. OPPONENT LOOKUP TABLE (Player vs Team)
# ---------------------------------------------------------
# NOTE: These averages are computed over ALL available matches for each player.
# During training, features like vs_opponent_runs_avg were computed as expanding
# historical averages (excluding the current match). At serving time we predict
# a FUTURE match so all past data is legitimately available, making this approach
# correct for production, though the values may differ slightly from training.
bat_vs_opp = batsman_df.groupby(['player', 'opponent'])[['runs', 'strike_rate']].mean().reset_index()
bat_vs_opp.rename(columns={'runs': 'vs_opponent_runs_avg', 'strike_rate': 'vs_opponent_strike_rate_avg'}, inplace=True)

bowl_vs_opp = bowler_df.groupby(['player', 'opponent'])[['wickets', 'economy']].mean().reset_index()
bowl_vs_opp.rename(columns={'wickets': 'vs_opponent_wickets_avg', 'economy': 'vs_opponent_economy_avg'}, inplace=True)

bat_vs_opp.to_csv('prod_bat_vs_opp.csv', index=False)
bowl_vs_opp.to_csv('prod_bowl_vs_opp.csv', index=False)
print("✅ Saved Opponent Lookups (Player vs Team)")

# ---------------------------------------------------------
# 3. VENUE LOOKUP TABLE (Player at Venue)
# ---------------------------------------------------------
bat_at_venue = batsman_df.groupby(['player', 'venue'])[['runs', 'strike_rate']].mean().reset_index()
bat_at_venue.rename(columns={'runs': 'venue_runs_avg', 'strike_rate': 'venue_strike_rate_avg'}, inplace=True)

bowl_at_venue = bowler_df.groupby(['player', 'venue'])[['wickets', 'economy']].mean().reset_index()
bowl_at_venue.rename(columns={'wickets': 'venue_wickets_avg', 'economy': 'venue_economy_avg'}, inplace=True)

bat_at_venue.to_csv('prod_bat_at_venue.csv', index=False)
bowl_at_venue.to_csv('prod_bowl_at_venue.csv', index=False)
print("✅ Saved Venue Lookups (Player at Venue)")

print("\n✨ Production Data Ready!")