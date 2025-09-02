import math, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .utils import wilder_atr, sma, rolling_median, quantize, sign

@dataclass
class Trade:
    side: int       # +1 long, -1 short
    entry_ts: int
    entry_px: float
    atr_entry: float
    r_value: float          # initial R in price units (ATR * sl_mult)
    sl_px: float            # current stop
    tsl_on_ts: int = 0
    last_tsl_step_ts: int = 0
    state: str = "OPEN"     # OPEN/CLOSED
    exit_ts: int = 0
    exit_px: float = 0.0
    exit_reason: str = ""   # TSL/SL/BE/TP
    R: float = 0.0
    overshoot_R: float = 0.0

class RiskManager:
    def __init__(self, cfg, symbol):
        self.cfg = cfg["risk"]
        self.tsl_cfg = self.cfg["tsl"]
        self.be_cfg = self.cfg["be"]
        self.server_cfg = self.cfg.get("server_stops", {"enabled": False, "min_replace_secs": 60})
        self.atr_win = self.cfg["atr"]["window"]
        self.atr_method = self.cfg["atr"]["method"]
        self.sl_mult = float(self.cfg["sl_mult"])
        self.tp_mult = float(self.cfg.get("tp_mult", 0.0))
        self.tsl_enabled = bool(self.tsl_cfg.get("enabled", True))
        self.be_enabled = bool(self.be_cfg.get("enabled", True))
        self.be_threshold_R = float(self.be_cfg.get("threshold_R", 0.5))
        self.tsl_atr_mult = float(self.tsl_cfg["tsl_atr_mult"])
        self.step_atr_mult = float(self.tsl_cfg["step_atr_mult"])
        self.first_step_delay = int(self.tsl_cfg["first_step_delay_secs"])
        self.min_step_secs = int(self.tsl_cfg["min_step_secs"])
        self.floor_from_med_mult = float(self.tsl_cfg["floor_from_med_mult"])
        self.quant_tick = float(self.tsl_cfg.get("quantize_tick_size", 0.0))

        symcfg = cfg.get("symbols_cfg", {}).get(symbol, {})
        self.sym_tick = float(symcfg.get("tick_size", self.quant_tick))

    def compute_atr(self, df):
        if self.atr_method == "wilder":
            return wilder_atr(df["high"], df["low"], df["close"], self.atr_win)
        else:
            return sma((df["high"]-df["low"]).abs(), self.atr_win)

    def open_trade(self, ts, px, side, atr_now, med_atr):
        r_val = atr_now * self.sl_mult
        if side == 1:
            sl = px - r_val
        else:
            sl = px + r_val
        sl = quantize(sl, self.sym_tick or self.quant_tick)
        t = Trade(side=side, entry_ts=ts, entry_px=px, atr_entry=atr_now, r_value=r_val, sl_px=sl)
        # TSL armed time is set at entry; we wait first_step_delay before stepping
        t.tsl_on_ts = ts
        t.last_tsl_step_ts = ts
        return t

    def _be_check(self, t: Trade, ts, px):
        if not self.be_enabled or t.state != "OPEN": return
        # Move to breakeven once unrealized >= threshold_R
        R_unreal = (px - t.entry_px) * t.side / t.r_value
        if R_unreal >= self.be_threshold_R:
            new_sl = quantize(t.entry_px, self.sym_tick or self.quant_tick)
            if (t.side == 1 and new_sl > t.sl_px) or (t.side == -1 and new_sl < t.sl_px):
                t.sl_px = new_sl
                t.last_tsl_step_ts = ts  # treat as a step to avoid immediate next step

    def _tsl_distance_now(self, atr_now, med_atr):
        floor_gap = self.floor_from_med_mult * med_atr
        target_gap = self.tsl_atr_mult * atr_now
        return max(target_gap, floor_gap)

    def _advance_required(self, t: Trade, atr_now):
        # How much further must price go to make next step?
        return self.step_atr_mult * atr_now

    def _eligible_for_step(self, t: Trade, ts):
        if not self.tsl_enabled: return False
        if (ts - t.tsl_on_ts) < self.first_step_delay: return False
        if (ts - t.last_tsl_step_ts) < self.min_step_secs: return False
        return True

    def maybe_step_trailing(self, t: Trade, ts, px, atr_now, med_atr):
        if not self._eligible_for_step(t, ts): return
        # Only step if price advanced enough from last step-anchor (approx via last SL)
        # We treat "advance needed" relative to current SL, not entry, to avoid micro-chop stepping.
        adv_needed = self._advance_required(t, atr_now)
        if t.side == 1:
            if (px - t.sl_px) < adv_needed: return
            new_sl = px - self._tsl_distance_now(atr_now, med_atr)
            if new_sl <= t.sl_px: return  # don't move backward
            t.sl_px = quantize(new_sl, self.sym_tick or self.quant_tick)
        else:
            if (t.sl_px - px) < adv_needed: return
            new_sl = px + self._tsl_distance_now(atr_now, med_atr)
            if new_sl >= t.sl_px: return
            t.sl_px = quantize(new_sl, self.sym_tick or self.quant_tick)
        t.last_tsl_step_ts = ts

    def check_exit(self, t: Trade, ts, bar_open, bar_high, bar_low, bar_close):
        # Priority: SL (incl BE/TSL). Optional TP if tp_mult > 0.
        exit_reason = None
        exit_px = None
        overshoot_R = 0.0

        # Stop hit inside bar
        if t.side == 1:
            if bar_low <= t.sl_px:
                exit_px = t.sl_px
                # overshoot = how far below SL the low printed (in R units; negative is bad)
                overshoot_R = (bar_low - t.sl_px) * t.side / t.r_value
                exit_reason = "TSL" if t.sl_px >= t.entry_px else "SL"
        else:
            if bar_high >= t.sl_px:
                exit_px = t.sl_px
                overshoot_R = (t.sl_px - bar_high) * t.side / t.r_value
                exit_reason = "TSL" if t.sl_px <= t.entry_px else "SL"

        # Optional TP
        if exit_px is None and self.tp_mult and self.tp_mult > 0:
            tp_px = t.entry_px + t.side * (self.tp_mult * t.atr_entry)
            if (t.side == 1 and bar_high >= tp_px) or (t.side == -1 and bar_low <= tp_px):
                exit_px = tp_px
                exit_reason = "TP"

        if exit_px is not None:
            t.state = "CLOSED"
            t.exit_ts = ts
            t.exit_px = exit_px
            t.exit_reason = exit_reason
            t.R = (t.exit_px - t.entry_px) * t.side / t.r_value
            t.overshoot_R = overshoot_R
            return True
        return False
