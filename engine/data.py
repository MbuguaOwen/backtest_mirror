import os
import pandas as pd
from .utils import ensure_datetime_index

def _one_month_path(inputs_dir, symbol, year_month):
    return os.path.join(inputs_dir, f"{symbol}-{year_month}.csv")

def load_months_ohlcv(inputs_dir, symbol, months):
    parts = []
    for ym in months:
        p = _one_month_path(inputs_dir, symbol, ym)
        if os.path.exists(p):
            df = pd.read_csv(p)
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # expected columns: timestamp, open, high, low, close, volume
    if "timestamp" not in df.columns:
        # try rename common alternatives
        for cand in ["time", "date", "datetime"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    df = ensure_datetime_index(df).sort_index()
    return df
