import numpy as np
import pandas as pd

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df.index.dtype, np.datetime64):
        if "timestamp" in df.columns:
            ts = df["timestamp"].values
        else:
            ts = df.index.values
        # Try epoch ms/seconds or ISO
        try:
            idx = pd.to_datetime(ts, unit="ms", utc=True)
        except Exception:
            try:
                idx = pd.to_datetime(ts, unit="s", utc=True)
            except Exception:
                idx = pd.to_datetime(ts, utc=True, errors="coerce")
        df = df.copy()
        df.index = idx
    return df

def wilder_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/window, adjust=False).mean()
    return atr

def sma(x, w):
    return x.rolling(w).mean()

def rolling_median(x, w):
    return x.rolling(w).median()

def zscore(x, w):
    m = x.rolling(w).mean()
    s = x.rolling(w).std(ddof=0)
    return (x - m) / (s.replace(0, np.nan))

def quantize(value, tick):
    if tick is None or tick <= 0:
        return value
    return np.floor(value / tick) * tick

def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)
