# engine/data.py
from __future__ import annotations
import os
import zipfile
from io import TextIOWrapper
from typing import List, Optional

import pandas as pd
from tqdm.auto import tqdm


# ---------- basic fs helpers ----------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ---------- candidate paths we support ----------
# Both flat and subfolder forms, CSV or ZIP:
#   inputs/ETHUSDT-ticks-2025-01.csv
#   inputs/ETHUSDT/ETHUSDT-ticks-2025-01.csv
#   inputs/ETHUSDT-ticks-2025-01.zip
#   inputs/ETHUSDT/ETHUSDT-ticks-2025-01.zip
# and OHLCV equivalents without "-ticks-"

def _candidate_tick_paths(inputs_dir: str, symbol: str, ym: str) -> List[str]:
    base1 = os.path.join(inputs_dir, f"{symbol}-ticks-{ym}.csv")
    base2 = os.path.join(inputs_dir, symbol, f"{symbol}-ticks-{ym}.csv")
    z1    = os.path.join(inputs_dir, f"{symbol}-ticks-{ym}.zip")
    z2    = os.path.join(inputs_dir, symbol, f"{symbol}-ticks-{ym}.zip")
    return [base1, base2, z1, z2]

def _candidate_ohlcv_paths(inputs_dir: str, symbol: str, ym: str) -> List[str]:
    base1 = os.path.join(inputs_dir, f"{symbol}-{ym}.csv")
    base2 = os.path.join(inputs_dir, symbol, f"{symbol}-{ym}.csv")
    z1    = os.path.join(inputs_dir, f"{symbol}-{ym}.zip")
    z2    = os.path.join(inputs_dir, symbol, f"{symbol}-{ym}.zip")
    return [base1, base2, z1, z2]


# ---------- CSV/ZIP reading (ticks) ----------

def _read_tick_csv_any(path: str) -> pd.DataFrame:
    """
    Read tick CSV or ZIP containing CSV with columns like:
      timestamp, price, qty/quantity/size/amount/volume, is_buyer_maker (optional)
    Robust to mixed types, spaces, capitalization, and missing maker flag.
    """

    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        # lower + strip
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        # alias maps
        ts_names   = ["timestamp", "time", "ts"]
        px_names   = ["price", "p"]
        qty_names  = ["qty", "quantity", "size", "amount", "vol", "volume"]
        maker_names= ["is_buyer_maker", "isbuyermaker", "is_maker", "buyer_is_maker", "buyer_maker", "ismaker", "isbuyer", "is_seller_maker"]

        def first_exist(cands):
            for c in cands:
                if c in df.columns:
                    return c
            return None

        ts_col   = first_exist(ts_names)
        px_col   = first_exist(px_names)
        qty_col  = first_exist(qty_names)
        mk_col   = first_exist(maker_names)

        if ts_col is None or px_col is None:
            raise ValueError(f"tick file missing required columns (have: {list(df.columns)})")

        out = pd.DataFrame()
        # timestamp -> to numeric if possible, else datetime parse
        ts_raw = df[ts_col]
        ts_num = pd.to_numeric(ts_raw, errors="coerce")
        if ts_num.notna().any():
            unit = "ms" if ts_num.max() > 10**11 else "s"
            out["ts"] = pd.to_datetime(ts_num.astype("Int64"), unit=unit, utc=True)
        else:
            # likely ISO strings
            out["ts"] = pd.to_datetime(ts_raw, utc=True, errors="coerce")

        # price
        out["price"] = pd.to_numeric(df[px_col], errors="coerce")

        # qty (default to 1.0 if absent)
        if qty_col is not None:
            out["qty"] = pd.to_numeric(df[qty_col], errors="coerce")
        else:
            out["qty"] = 1.0

        # maker flag default 0
        if mk_col is not None:
            out["is_buyer_maker"] = pd.to_numeric(df[mk_col], errors="coerce").fillna(0).astype("int8")
        else:
            out["is_buyer_maker"] = 0

        # drop rows where ts/price/qty invalid
        out = out.loc[out["ts"].notna()]
        out = out.loc[out["price"].notna()]
        out = out.loc[out["qty"].notna()]
        return out.reset_index(drop=True)

    def _read_csv(path_or_io) -> pd.DataFrame:
        # Read fully, then normalize (don’t use usecols to avoid “Usecols do not match”)
        df = pd.read_csv(
            path_or_io,
            header=0,
            sep=None,            # auto-detect delimiter (keeps engine='python')
            engine="python",
            on_bad_lines="skip",
        )
        df = _normalize_columns(df)
        # small log to help diagnose mixed headers
        try:
            from tqdm.auto import tqdm as _tqdm
            _tqdm.write(f"[data] columns normalized -> {list(df.columns)} (rows={len(df)})")
        except Exception:
            pass
        return df[["ts", "price", "qty", "is_buyer_maker"]]

    if path.endswith(".csv"):
        return _read_csv(path)

    if path.endswith(".zip"):
        import zipfile
        from io import TextIOWrapper
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError(f"No CSV found inside {path}")
            with zf.open(csv_names[0], "r") as fh:
                return _read_csv(TextIOWrapper(fh, encoding="utf-8"))

    raise ValueError(f"Unsupported tick file: {path}")


# ---------- aggregation: ticks -> 1m OHLCV ----------

def _aggregate_ticks_to_1m_ohlcv(ticks: pd.DataFrame) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume","trades"])

    ticks = ticks.copy()
    # 'T' is deprecated; use 'min'
    ticks["minute"] = ticks["ts"].dt.floor("min")
    g = ticks.groupby("minute", sort=True)

    out = pd.DataFrame({
        "ts":     g["ts"].first().index,
        "open":   g["price"].first().values,
        "high":   g["price"].max().values,
        "low":    g["price"].min().values,
        "close":  g["price"].last().values,
        "volume": g["qty"].sum().values,
        "trades": g["price"].size().values,
    })
    return out.reset_index(drop=True)


# ---------- cache helpers (Parquet) ----------

def _cache_path(cache_dir: str, symbol: str, ym: str) -> str:
    return os.path.join(cache_dir, symbol, f"{symbol}-{ym}.parquet")

def _read_cached_ohlcv(cache_dir: str, symbol: str, ym: str) -> Optional[pd.DataFrame]:
    p = _cache_path(cache_dir, symbol, ym)
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None

def _write_cached_ohlcv(cache_dir: str, symbol: str, ym: str, df: pd.DataFrame) -> None:
    _ensure_dir(os.path.join(cache_dir, symbol))
    df.to_parquet(_cache_path(cache_dir, symbol, ym), index=False)


# ---------- public API ----------

def load_months_ohlcv(
    inputs_dir: str,
    symbol: str,
    months: List[str],
    cache_dir: str = os.path.join("inputs", "_ohlcv_cache"),
) -> pd.DataFrame:
    """
    Return concatenated 1m OHLCV for requested months.
    Preference per month:
      1) cached Parquet under inputs/_ohlcv_cache/<SYMBOL>/<SYMBOL>-YYYY-MM.parquet
      2) aggregate from ticks if tick CSV/ZIP exists
      3) use prebuilt OHLCV CSV/ZIP if present
    Raise FileNotFoundError if none exist for a requested month.
    """
    _ensure_dir(os.path.join(cache_dir, symbol))
    frames: List[pd.DataFrame] = []

    for ym in tqdm(months, desc=f"{symbol} data", unit="mo", leave=False):
        # 1) cache
        cached = _read_cached_ohlcv(cache_dir, symbol, ym)
        if cached is not None and not cached.empty:
            frames.append(cached)
            continue

        # 2) ticks
        tick_path = _find_first_existing(_candidate_tick_paths(inputs_dir, symbol, ym))
        if tick_path:
            tqdm.write(f"[data] {symbol} {ym}: aggregating ticks -> 1m from {os.path.relpath(tick_path)}")
            ticks = _read_tick_csv_any(tick_path)
            ohlcv = _aggregate_ticks_to_1m_ohlcv(ticks)
            _write_cached_ohlcv(cache_dir, symbol, ym, ohlcv)
            frames.append(ohlcv)
            continue

        # 3) prebuilt OHLCV
        ohlcv_path = _find_first_existing(_candidate_ohlcv_paths(inputs_dir, symbol, ym))
        if ohlcv_path:
            tqdm.write(f"[data] {symbol} {ym}: using prebuilt OHLCV {os.path.relpath(ohlcv_path)}")
            if ohlcv_path.endswith(".csv"):
                df = pd.read_csv(ohlcv_path)
            else:
                with zipfile.ZipFile(ohlcv_path) as zf:
                    csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if not csv_names:
                        raise FileNotFoundError(f"No CSV found inside {ohlcv_path}")
                    with zf.open(csv_names[0], "r") as fh:
                        df = pd.read_csv(TextIOWrapper(fh, encoding="utf-8"))
            if "ts" not in df.columns:
                if "timestamp" in df.columns:
                    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                else:
                    raise ValueError(f"OHLCV file {ohlcv_path} missing 'ts' column")
            try:
                _write_cached_ohlcv(cache_dir, symbol, ym, df)
            except Exception as e:
                tqdm.write(f"[data] cache write skipped: {e}")
            frames.append(df)
            continue

        raise FileNotFoundError(
            f"No data found for {symbol} {ym} under {inputs_dir}. "
            "Expected tick CSV/ZIP or OHLCV CSV/ZIP."
        )

    if not frames:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume","trades"])

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="first")
    return df.reset_index(drop=True)
