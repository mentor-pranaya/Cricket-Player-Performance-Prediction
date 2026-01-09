import pandas as pd

def clean_data():
    # Load raw data
    matches = pd.read_csv("data/raw/matches.csv")
    balls = pd.read_csv("data/raw/deliveries.csv")

    print("Files loaded successfully")

    # Print columns (DEBUG)
    print("Matches columns:", matches.columns)
    print("Balls columns:", balls.columns)

    # Handle column name differences safely
    batter_col = 'batter' if 'batter' in balls.columns else 'batsman'
    bowler_col = 'bowler'

    # Drop missing values
    matches = matches.dropna(subset=['venue'])
    balls = balls.dropna(subset=[batter_col, bowler_col])

    # Normalize text columns
    matches['venue'] = matches['venue'].str.lower()
    balls[batter_col] = balls[batter_col].str.lower()
    balls[bowler_col] = balls[bowler_col].str.lower()

    # Handle date column safely
    if 'date' in matches.columns:
        matches['date'] = pd.to_datetime(matches['date'])
    elif 'match_date' in matches.columns:
        matches['match_date'] = pd.to_datetime(matches['match_date'])

    # Save cleaned files
    matches.to_csv("data/raw/matches_cleaned.csv", index=False)
    balls.to_csv("data/raw/deliveries_cleaned.csv", index=False)

    print(" Data cleaning completed successfully")

if __name__ == "__main__":
    clean_data()
    
    

