import pandas as pd

matches = pd.read_csv('DATA1/matches.csv')
deliveries = pd.read_csv('DATA1/deliveries.csv')

# Clean matches
matches['city'].fillna('Unknown', inplace=True)
matches['player_of_match'].fillna('Unknown', inplace=True)
matches['winner'].fillna('No Result', inplace=True)
matches['result_margin'].fillna(0, inplace=True)
matches['target_runs'].fillna(0, inplace=True)
matches['target_overs'].fillna(0, inplace=True)
matches.drop(columns=['method'], inplace=True)

# Clean deliveries
deliveries['extras_type'].fillna('None', inplace=True)
deliveries['player_dismissed'].fillna('None', inplace=True)
deliveries['dismissal_kind'].fillna('None', inplace=True)
deliveries['fielder'].fillna('None', inplace=True)

# Save cleaned files
matches.to_csv('DATA1/matches_cleaned.csv', index=False)
deliveries.to_csv('DATA1/deliveries_cleaned.csv', index=False)
