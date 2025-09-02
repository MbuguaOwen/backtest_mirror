import pandas as pd

class Trigger:
    """
    Two modes:
      - donchian: breakout above/below rolling extremes
      - zscore: sign of zscore release (already computed in EventWave)
    """
    def __init__(self, cfg):
        self.mode = cfg["trigger"]["mode"]
        self.dc_win = cfg["trigger"]["donchian_window"]
        self.z_win = cfg["trigger"]["zscore_window"]
        self.z_th = cfg["trigger"]["zscore_threshold"]

    def compute_donchian(self, high, low):
        dc_high = high.rolling(self.dc_win).max()
        dc_low = low.rolling(self.dc_win).min()
        return dc_high, dc_low

    def signal(self, df, release_up, release_dn):
        if self.mode == "donchian":
            hi, lo = self.compute_donchian(df["high"], df["low"])
            long_sig = (df["close"] > hi.shift(1)).astype(int)
            short_sig = (df["close"] < lo.shift(1)).astype(int)
        else:
            # zscore mode piggybacks release flags
            long_sig = release_up
            short_sig = release_dn
        return long_sig, short_sig
