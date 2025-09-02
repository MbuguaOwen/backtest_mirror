import os, json, yaml, argparse, re, datetime as dt
import math
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from engine.data import load_months_ohlcv
from engine.backtest import BacktestEngine  # keep your actual path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--walkforward", action="store_true", help="force walk-forward mode")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbols = cfg.get("symbols", ["BTCUSDT"])
    inputs_dir = cfg.get("paths", {}).get("inputs_dir", "inputs")
    outputs_dir = cfg.get("paths", {}).get("outputs_dir", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    months = cfg.get("data", {}).get("months")
    if not months or not isinstance(months, list):
        months = _infer_months(inputs_dir)

    # Backtest mode selection
    bt_cfg = cfg.get("backtest", {})
    mode = "walkforward" if args.walkforward else bt_cfg.get("mode", "single")

    if mode == "walkforward":
        wf = bt_cfg.get("walkforward", {})
        train_m = int(wf.get("train_months", 3))
        test_m = int(wf.get("test_months", 1))
        step_m = int(wf.get("step_months", 1))
        start = wf.get("start")
        end = wf.get("end")
        months_use = _filter_months(months, start, end)
        tqdm.write(
            f"[backtest] mode=walkforward months={months_use} train={train_m} test={test_m} step={step_m}"
        )
        _run_walkforward(cfg, symbols, inputs_dir, outputs_dir, months_use, train_m, test_m, step_m)
        return

    tqdm.write(f"[backtest] mode=single symbols={symbols} inputs={inputs_dir} months={months}")

    # Overall accumulators for final recap
    total_trades = 0
    total_sum_R: float = 0.0
    total_wr_n = 0
    total_wr_k = 0  # count of trades used for WR calculation

    # ---- SYMBOLS progress bar ----
    for sym in tqdm(symbols, desc="SYMBOLS", unit="sym"):
        tqdm.write(f"[backtest] Loading {sym} months={months}")
        df = load_months_ohlcv(inputs_dir, sym, months)

        if df.empty:
            tqdm.write(f"[warn] No data for {sym}; skipping")
            continue

        engine = BacktestEngine(cfg, sym)
        trades, audit = engine.run(df)

        sym_dir = os.path.join(outputs_dir, sym)
        os.makedirs(sym_dir, exist_ok=True)
        trades_csv = os.path.join(sym_dir, "trades.csv")
        summary_json = os.path.join(sym_dir, "summary.json")

        try:
            trades.to_csv(trades_csv, index=False)
        except Exception as e:
            tqdm.write(f"[warn] write trades.csv failed: {e}")

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(audit.get("summary", audit), f, indent=2, default=str)

        tqdm.write(f"[ok] {sym} wrote {trades_csv}")
        tqdm.write(f"[ok] {sym} wrote {summary_json}")

        # ---- Recap for this symbol ----
        sum_R, win_rate, used_for_wr = _compute_recap_metrics(trades, audit)
        total_trades += len(trades)
        total_sum_R += sum_R
        total_wr_n += int(round(win_rate * used_for_wr)) if used_for_wr else 0
        total_wr_k += used_for_wr
        if math.isnan(win_rate):
            tqdm.write(f"[recap] {sym}: trades={len(trades)} sum_R={sum_R:.4f} (win_rate: n/a)")
        else:
            tqdm.write(
                f"[recap] {sym}: trades={len(trades)} win_rate={win_rate:.2%} sum_R={sum_R:.4f}"
            )

    # ---- Overall recap ----
    overall_wr = (total_wr_n / total_wr_k) if total_wr_k else float("nan")
    if math.isnan(overall_wr):
        tqdm.write(f"[recap] ALL: trades={total_trades} sum_R={total_sum_R:.4f}")
    else:
        tqdm.write(
            f"[recap] ALL: trades={total_trades} win_rate={overall_wr:.2%} sum_R={total_sum_R:.4f}"
        )


def _infer_months(inputs_dir: str) -> list[str]:
    rx = re.compile(r"(\d{4})-(\d{1,2})")
    seen = set()
    for root, _dirs, files in os.walk(inputs_dir):
        for fn in files:
            m = rx.search(fn)
            if m:
                y, mo = int(m.group(1)), int(m.group(2))
                seen.add(f"{y:04d}-{mo:02d}")
    months = sorted(seen)
    if not months:
        raise SystemExit(
            f"No data files found under {inputs_dir}. Place tick CSVs like ETHUSDT-ticks-2025-01.csv"
        )
    return months


def _filter_months(months: list[str], start: Optional[str], end: Optional[str]) -> list[str]:
    def to_key(m: str):
        y, mm = m.split("-")
        return (int(y), int(mm))

    items = sorted(months, key=to_key)
    if start:
        items = [m for m in items if to_key(m) >= to_key(start)]
    if end:
        items = [m for m in items if to_key(m) <= to_key(end)]
    return items


def _month_add(m: str, k: int) -> str:
    y, mm = map(int, m.split("-"))
    total = y * 12 + (mm - 1) + k
    ny, nmo = divmod(total, 12)
    return f"{ny:04d}-{nmo+1:02d}"


def _make_folds(months: list[str], train_m: int, test_m: int, step_m: int):
    # sliding window folds over months
    folds = []
    for i in range(0, len(months) - train_m - test_m + 1, step_m):
        train = months[i : i + train_m]
        test = months[i + train_m : i + train_m + test_m]
        folds.append((train, test))
    return folds


def _run_walkforward(cfg, symbols, inputs_dir, outputs_dir, months, train_m, test_m, step_m):
    folds = _make_folds(months, train_m, test_m, step_m)
    tqdm.write(f"[walkforward] folds={len(folds)}")
    all_tot_trades = 0
    all_sum_R = 0.0
    all_wr_n = 0
    all_wr_k = 0
    for fi, (train, test) in enumerate(folds):
        tqdm.write(f"[fold {fi}] train={train} test={test}")
        # Load train+test for indicator continuity…
        both = train + test
        for sym in tqdm(symbols, desc=f"FOLD{fi}", unit="sym", leave=False):
            df = load_months_ohlcv(inputs_dir, sym, both)
            if df.empty:
                tqdm.write(f"[warn] {sym} empty in fold {fi}")
                continue
            engine = BacktestEngine(cfg, sym)
            trades, audit = engine.run(df)
            # Keep only trades whose entry_ts falls inside the TEST window
            test_start = test[0]
            test_end = test[-1]
            # convert YYYY-MM to [start_ts, next_month_start_ts)
            ts0 = pd.to_datetime(test_start + "-01", utc=True)
            ts1 = pd.to_datetime(_month_add(test_end, 1) + "-01", utc=True)
            e_ts = pd.to_datetime(trades["entry_ts"], unit="s", utc=True)
            mask = (e_ts >= ts0) & (e_ts < ts1)
            trades = trades.loc[mask].reset_index(drop=True)

            sym_dir = os.path.join(outputs_dir, f"{sym}_wf")
            os.makedirs(sym_dir, exist_ok=True)
            trades_csv = os.path.join(sym_dir, f"fold{fi}_trades.csv")
            summary_json = os.path.join(sym_dir, f"fold{fi}_summary.json")
            try:
                trades.to_csv(trades_csv, index=False)
            except Exception as e:
                tqdm.write(f"[warn] write {trades_csv} failed: {e}")
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump({"fold": fi, "train": train, "test": test}, f, indent=2)

            sum_R, win_rate, used_for_wr = _compute_recap_metrics(trades, audit)
            all_tot_trades += len(trades)
            all_sum_R += sum_R
            all_wr_n += int(round(win_rate * used_for_wr)) if used_for_wr else 0
            all_wr_k += used_for_wr
            if math.isnan(win_rate):
                tqdm.write(
                    f"[recap fold {fi}] {sym}: trades={len(trades)} sum_R={sum_R:.4f} (win_rate: n/a)"
                )
            else:
                tqdm.write(
                    f"[recap fold {fi}] {sym}: trades={len(trades)} win_rate={win_rate:.2%} sum_R={sum_R:.4f}"
                )

    overall_wr = (all_wr_n / all_wr_k) if all_wr_k else float("nan")
    if math.isnan(overall_wr):
        tqdm.write(f"[WF ALL] trades={all_tot_trades} sum_R={all_sum_R:.4f}")
    else:
        tqdm.write(
            f"[WF ALL] trades={all_tot_trades} win_rate={overall_wr:.2%} sum_R={all_sum_R:.4f}"
        )


def _compute_recap_metrics(trades, audit) -> tuple[float, float, int]:
    """
    Returns (sum_R, win_rate, used_for_wr_count).
    Tries to read 'R' (or common variants) from trades; falls back to audit summary if needed.
    win_rate is computed as mean(R>0) over trades that have an R column.
    """
    r_col: Optional[str] = None
    for c in ["R", "r", "realized_R", "realized_r"]:
        if c in trades.columns:
            r_col = c
            break

    if r_col is not None and len(trades) > 0:
        r = trades[r_col]
        r_valid = r.dropna()
        sum_R = float(r_valid.sum())
        used_for_wr = int(r_valid.shape[0])
        win_rate = float((r_valid > 0).mean()) if used_for_wr else float("nan")
        return sum_R, win_rate, used_for_wr

    # Fallback: try audit summary
    summary = audit.get("summary", audit) if isinstance(audit, dict) else {}
    sum_R = float(summary.get("sum_R", 0.0))
    win_rate = float(summary.get("win_rate", float("nan")))
    return sum_R, win_rate, 0


if __name__ == "__main__":
    main()

