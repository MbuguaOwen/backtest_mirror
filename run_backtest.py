#!/usr/bin/env python3
import argparse, os, json, yaml, time
import pandas as pd
from engine.data import load_months_ohlcv
from engine.backtest import BacktestEngine
from engine.metrics import summarize_performance

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    inputs_dir = cfg["paths"]["inputs_dir"]
    outputs_dir = cfg["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    symbols = cfg["symbols"]
    months = cfg["months"]

    for sym in symbols:
        print(f"[backtest] Loading {sym} months={months}")
        df = load_months_ohlcv(inputs_dir, sym, months)
        if df.empty:
            raise SystemExit(f"No data found for {sym} under {inputs_dir} for months={months}")

        print(f"[backtest] Running engine for {sym} bars={len(df)} ...")
        engine = BacktestEngine(cfg, sym)
        trades, audit = engine.run(df)

        trades_csv = os.path.join(outputs_dir, f"trades_{sym}.csv")
        trades.to_csv(trades_csv, index=False)
        summary = summarize_performance(trades, audit)
        summary_json = os.path.join(outputs_dir, f"summary_{sym}.json")
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== SUMMARY {sym} ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print(f"\n[ok] Wrote {trades_csv}\n[ok] Wrote {summary_json}")

if __name__ == "__main__":
    main()
