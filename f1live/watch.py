"""Helpers the live dashboards share for text tables and predictions.csv."""
import pandas as pd


def fmt(v, digits=3):
    """Table cell: '-' for missing, floats rounded, everything else as is."""
    if pd.isna(v):
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)


def bar(p, width=30):
    return "█" * round(p * width) + "░" * (width - round(p * width))


def table(df, cols=None, digits=3):
    """Right-aligned plain-text table."""
    cols = list(cols or df.columns)
    cells = [[fmt(v, digits) for v in row] for row in df[cols].itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in cells)) if cells else len(c) for i, c in enumerate(cols)]
    lines = ["  ".join(c.rjust(w) for c, w in zip(cols, widths))]
    return lines + ["  ".join(v.rjust(w) for v, w in zip(r, widths)) for r in cells]


TEXT_COLUMNS = {"drivers": str, "numbers": str, "gaps": str, "laps": str, "note": str, "plan": str}   # lists written as 'a b c'


def read(path):
    """predictions.csv -> DataFrame, or None if missing / empty / caught mid-write. Columns that hold space-separated lists
    are read as text: with only one-item lists so far pandas would turn them into numbers."""
    try:
        df = pd.read_csv(path, dtype=TEXT_COLUMNS)
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError):
        return None
    return df if len(df) else None
