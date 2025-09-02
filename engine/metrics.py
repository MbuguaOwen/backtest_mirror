import pandas as pd
import numpy as np


def summarize_performance(trades: pd.DataFrame, audit: dict):
    if trades.empty:
        return {"trades": 0}

    def sum_where(col, cond):
        return trades.loc[cond, col].sum()

    tsl = trades["exit_reason"] == "TSL"
    sl = trades["exit_reason"] == "SL"
    tp = trades["exit_reason"] == "TP"
    # Treat BE as TSL whose final stop equals entry price (within tiny epsilon)
    eps = 1e-9
    be = tsl & ((trades["sl_px_final"] - trades["entry_px"]).abs() <= eps)

    out = {
        "trades": int(len(trades)),
        "TSL": int(tsl.sum()),
        "SL": int(sl.sum()),
        "TP": int(tp.sum()),
        "BE": int(be.sum()),
        "sum_r_tsl": float(sum_where("R", tsl)),
        "sum_r_sl": float(sum_where("R", sl)),
        "sum_r_tp": float(sum_where("R", tp)),
        "sum_r_be": float(sum_where("R", be)),
        "sum_r_total": float(trades["R"].sum()),
        "avg_R": float(trades["R"].mean()),
        "median_R": float(trades["R"].median()),
        "win_rate": float((trades["R"] > 0).mean()),
        "sum_r_sl_overshoot": float(trades.loc[sl, "overshoot_R"].sum()),
        "audit": audit,
    }
    return out

