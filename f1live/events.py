"""Loading events.csv into DataFrames with consistent dtypes."""
import pandas as pd


def prepare(df):
    """ts -> datetime, driver -> car number as str ('44') or None for rows without a driver."""
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    df["driver"] = df["driver"].map(lambda x: str(int(x)) if pd.notna(x) else None)
    return df


def load(path):
    return prepare(pd.read_csv(path))
