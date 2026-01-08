import pandas as pd
import os

# ===== PATHS (MATCH YOUR CURRENT STRUCTURE) =====
RAW_PATH = "data/raw/raw/"
CLEAN_PATH = "data/cleaned/"

os.makedirs(CLEAN_PATH, exist_ok=True)

# ===== LOAD DATA =====
matches = pd.read_csv(RAW_PATH + "matches.csv")
deliveries = pd.read_csv(RAW_PATH + "deliveries.csv")

# ===== STANDARDIZE COLUMN NAMES =====
matches.columns = matches.columns.str.lower().str.replace(" ", "_")
deliveries.columns = deliveries.columns.str.lower().str.replace(" ", "_")

# ===== DATE CONVERSION =====
matches['date'] = pd.to_datetime(matches['date'], errors='coerce')

# ===== HANDLE MISSING VALUES =====
matches.dropna(subset=['id'], inplace=True)
deliveries.dropna(subset=['match_id'], inplace=True)

# ===== CREATE is_wicket COLUMN (CORRECT WAY) =====
# If player_dismissed is NOT null → wicket happened
deliveries['is_wicket'] = deliveries['player_dismissed'].notna().astype(int)

# ===== HANDLE RUN COLUMNS =====
deliveries['total_runs'] = deliveries['total_runs'].fillna(0)
deliveries['batsman_runs'] = deliveries['batsman_runs'].fillna(0)

# ===== REMOVE DUPLICATES =====
matches.drop_duplicates(inplace=True)
deliveries.drop_duplicates(inplace=True)

# ===== SAVE CLEANED FILES =====
matches.to_csv(CLEAN_PATH + "matches_cleaned.csv", index=False)
deliveries.to_csv(CLEAN_PATH + "deliveries_cleaned.csv", index=False)

print("✅ DATA CLEANING COMPLETED SUCCESSFULLY")
