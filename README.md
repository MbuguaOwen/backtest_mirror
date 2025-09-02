# Backtest Mirror — TSMOM → EventWave → Trigger → Entry → Risk

A pragmatic backtest harness that mirrors a live engine:
- **Regime (TSMOM)** across multiple lookback windows (1m series; lookbacks act as minutes).
- **EventWave (Squeeze → Release)** via rolling ATR compression and z‑score ignition.
- **Trigger** via Donchian/Z breakout aligned to regime direction.
- **Entry** at next bar (market-on-open of the next candle).
- **Risk** with ATR‑based SL/TP, Breakeven, and *stepwise Trailing Stop (TSL)* featuring:
  - First‑step delay (to prevent instant trail on ignition).
  - Min time between steps (debounce).
  - ATR‑floor so trail distance can’t collapse after compression.
  - Tick quantization + server‑stop emulation (debounced).

> This mirrors a live-style pipeline without external deps (Binance, WS). It’s bar‑by‑bar, no look‑ahead.

## Data format

Place CSVs under `inputs/` named like `BTCUSDT-2025-01.csv`.
Required columns: `timestamp,open,high,low,close,volume`.
- `timestamp` ISO8601 or integer epoch ms/seconds (auto-detected).

## Run

```bash
pip install pandas pyyaml numpy
python run_backtest.py --config configs/default.yaml
```

Outputs:
- `outputs/trades_<SYMBOL>.csv` — all trades
- `outputs/summary_<SYMBOL>.json` — metrics
- Console summary — TSL/BE/SL counts, R buckets, overshoot diagnostics

## Notes on TSL logic

TSL activates only after **first_step_delay_secs**.
It then steps no more frequently than **min_step_secs**.
Trail distance = `max(tsl_atr_mult * ATR, floor_from_med_mult * median_ATR)`.
Stops are **quantized** to `quantize_tick_size` (per‑symbol) to mirror exchange behavior.
Overshoot is measured in **R** (distance beyond SL at stop bar / initial R).

## Caveats

- This is 1‑minute bar simulation. If your live engine aggregates ticks, align your preprocessing.
- For multi‑TF regime, we apply “minutes as lookback length”; it’s a robust approximation.
- If you need perfect parity, port your exact functions into the `engine/` modules (names align).

