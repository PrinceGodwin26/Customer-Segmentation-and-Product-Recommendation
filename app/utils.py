import pandas as pd


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a dataframe to JSON-safe records (handles pandas NA/Int64)."""
    if df is None or df.empty:
        return []
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
