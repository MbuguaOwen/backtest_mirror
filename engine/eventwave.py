import pandas as pd
from .utils import wilder_atr, rolling_median, zscore

class EventWave:
    """
    Squeeze: ATR < pct_of_median * median_ATR over window.
    Release: price zscore over zscore_window > threshold (up) or < -threshold (down).
    """
    def __init__(self, cfg):
        s = cfg["eventwave"]["squeeze"]
        r = cfg["eventwave"]["release"]
        self.atr_window = s["atr_window"]
        self.pct_of_median = s["pct_of_median"]
        self.median_window = s["median_window"]
        self.zscore_window = r["zscore_window"]
        self.zscore_threshold = r["zscore_threshold"]

    def compute(self, df):
        atr = wilder_atr(df["high"], df["low"], df["close"], self.atr_window)
        med = rolling_median(atr, self.median_window)
        squeeze = (atr < self.pct_of_median * med).astype(int)

        z = zscore(df["close"], self.zscore_window)
        release_up = (z > self.zscore_threshold).astype(int)
        release_dn = (z < -self.zscore_threshold).astype(int)
        return squeeze, release_up, release_dn, atr, med
