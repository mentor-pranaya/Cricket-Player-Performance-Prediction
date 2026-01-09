import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR.parent / "data" / "raw"
PROCESSED_PATH = BASE_DIR.parent / "data" / "processed"

def load_data():
    matches = pd.read_csv(RAW_PATH / "ipl_matches_data.csv")
    balls = pd.read_csv(RAW_PATH / "ball_by_ball_data.csv")
    return matches, balls

def clean_matches(matches):
    matches.columns = matches.columns.str.lower()
    matches['match_date'] = pd.to_datetime(matches['match_date'], errors='coerce')
    return matches

def clean_balls(balls):
    balls.columns = balls.columns.str.lower()
    balls = balls.drop_duplicates()

    # Fill missing numeric values
    numeric_cols = balls.select_dtypes(include=np.number).columns
    balls[numeric_cols] = balls[numeric_cols].fillna(0)

    # Fill categorical missing values
    if 'player_dismissed' in balls.columns:
        balls['player_dismissed'] = balls['player_dismissed'].fillna("Not Out")

    return balls

def merge_data(matches, balls):
    df = balls.merge(matches, on="match_id", how="left")
    return df

def save_data(df):
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH / "cleaned_data.csv", index=False)

def main():
    matches, balls = load_data()
    matches = clean_matches(matches)
    balls = clean_balls(balls)
    df = merge_data(matches, balls)
    save_data(df)
    print("✅ Cleaned data saved to data/processed/cleaned_data.csv")

if __name__ == "__main__":
    main()
