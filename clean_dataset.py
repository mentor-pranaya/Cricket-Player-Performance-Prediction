import pandas as pd


ipl = pd.read_csv(r"D:/cricket/IPL.csv", low_memory=False)
batting = pd.read_csv(r"D:/cricket/alldatabatting.csv")
bowling = pd.read_csv(r"D:/cricket/alldatabowling.csv")


def clean_columns(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

ipl = clean_columns(ipl)
batting = clean_columns(batting)
bowling = clean_columns(bowling)


ipl["date"] = pd.to_datetime(ipl["date"], errors="coerce")


merged_df = ipl.merge(
    batting,
    how="left",
    left_on="batter",
    right_on="name"
)


merged_df = merged_df.merge(
    bowling,
    how="left",
    left_on="bowler",
    right_on="name",
    suffixes=("_bat", "_bowl")
)


merged_df.drop(columns=["name_bat", "name_bowl"], errors="ignore", inplace=True)


numeric_cols = merged_df.select_dtypes(include="number").columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

categorical_cols = merged_df.select_dtypes(include="object").columns
merged_df[categorical_cols] = merged_df[categorical_cols].fillna("Unknown")


merged_df.to_csv("merged_clean_ipl.csv", index=False)

print("Merged and cleaned dataset saved as merged_clean_ipl.csv")
