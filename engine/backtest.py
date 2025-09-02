import time
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from .regime import RegimeClassifier
from .eventwave import EventWave
from .trigger import Trigger
from .entry import EntryManager
from .risk import RiskManager, Trade


class BacktestEngine:
    def __init__(self, cfg, symbol):
        self.cfg = cfg
        self.symbol = symbol
        self.warmup_bars = int(cfg["engine"]["warmup"]["min_1m_bars"])

        self.regime = RegimeClassifier(cfg)
        self.wave = EventWave(cfg)
        self.trig = Trigger(cfg)
        self.entry = EntryManager(cfg)
        self.risk = RiskManager(cfg, symbol)

        self.cooldown_bars = int(cfg["entry"]["cooldown_bars"])

    def _epoch(self, ts):
        # ts is pandas Timestamp (UTC)
        return int(ts.timestamp())

    def run(self, df):
        df = df.copy()
        # Ensure datetime index from 'ts' column if provided; else coerce index
        if "ts" in df.columns:
            try:
                df.index = pd.to_datetime(df["ts"], utc=True)
                df.drop(columns=["ts"], inplace=True)
            except Exception:
                pass
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.sort_index(inplace=True)
        # Precompute wave/regime/atr
        squeeze, rel_up, rel_dn, atr, med_atr = self.wave.compute(df)
        regime = self.regime.compute(df["close"])
        long_trig, short_trig = self.trig.signal(df, rel_up, rel_dn)

        df["atr"] = atr
        df["med_atr"] = med_atr
        df["squeeze"] = squeeze
        df["rel_up"] = rel_up
        df["rel_dn"] = rel_dn
        df["regime"] = regime
        df["long_trig"] = long_trig
        df["short_trig"] = short_trig

        trades = []
        open_trade = None
        cooldown = 0
        last_fill_signal = None

        # Walk bars
        for ts, row in tqdm(df.iterrows(), total=len(df), desc=self.symbol, unit="bar", leave=True):
            i_epoch = self._epoch(ts)

            # Skip until warmup complete
            if len(df.loc[:ts]) < self.warmup_bars:
                continue

            # Per-bar cooldown
            self.entry.tick()
            if cooldown > 0:
                cooldown -= 1

            # If open trade, manage risk
            if open_trade is not None:
                # BE step (doesn't advance last_tsl_step unless moved)
                self.risk._be_check(open_trade, i_epoch, row["close"])
                # TSL step
                self.risk.maybe_step_trailing(open_trade, i_epoch, row["close"], row["atr"], row["med_atr"])
                # Exit check
                exited = self.risk.check_exit(open_trade, i_epoch, row["open"], row["high"], row["low"], row["close"])
                if exited:
                    trades.append(open_trade)
                    open_trade = None
                    cooldown = self.cooldown_bars
                    continue

            # If flat and not cooling down, check entries
            if open_trade is None and cooldown == 0 and self.entry.ready():
                # Align direction with regime
                can_long = self.entry.can_long() and row["regime"] == 1
                can_short = self.entry.can_short() and row["regime"] == -1

                long_ok = can_long and (row["squeeze"] == 1) and (row["long_trig"] == 1)
                short_ok = can_short and (row["squeeze"] == 1) and (row["short_trig"] == 1)

                if long_ok or short_ok:
                    side = 1 if long_ok else -1
                    # Fill at next bar open - we approximate using current open for simplicity,
                    # but we can lag by 1 bar if strict.
                    fill_px = row["open"]
                    open_trade = self.risk.open_trade(self._epoch(ts), fill_px, side, row["atr"], row["med_atr"])
                    self.entry.arm_cooldown()
                    last_fill_signal = "LONG" if side == 1 else "SHORT"
                    continue

        # Close any leftover (mark-to-close)
        if open_trade is not None:
            open_trade.state = "CLOSED"
            open_trade.exit_ts = self._epoch(df.index[-1])
            open_trade.exit_px = df["close"].iloc[-1]
            open_trade.exit_reason = "CLOSE"
            open_trade.R = (open_trade.exit_px - open_trade.entry_px) * open_trade.side / open_trade.r_value
            trades.append(open_trade)

        # Build trades frame
        if trades:
            recs = [{
                "side": t.side,
                "entry_ts": t.entry_ts,
                "entry_px": t.entry_px,
                "exit_ts": t.exit_ts,
                "exit_px": t.exit_px,
                "exit_reason": t.exit_reason,
                "R": t.R,
                "overshoot_R": t.overshoot_R,
                "sl_px_final": t.sl_px,
            } for t in trades]
            trades_df = pd.DataFrame(recs)
        else:
            trades_df = pd.DataFrame(columns=["side","entry_ts","entry_px","exit_ts","exit_px","exit_reason","R","overshoot_R","sl_px_final"])

        audit = {
            "bars_seen": int(len(df)),
            "warmup_bars": int(self.warmup_bars),
            "entries": int((trades_df["R"].notna()).sum()),
            "config_snapshot": self.cfg,
        }
        return trades_df, audit

