import numpy as np
import pandas as pd

class RegimeClassifier:
    """
    Simple TSMOM regime: for each timeframe (interpreted via lookback_closes on 1m bars),
    direction = sign(close - close.shift(lookback)). Majority vote sets regime.
    """
    def __init__(self, cfg):
        self.tfs = cfg["regime"]["ts_mom"]["timeframes"]
        self.req = cfg["regime"]["ts_mom"]["require_majority"]

    def compute(self, close: pd.Series):
        votes = []
        for tf in self.tfs:
            lb = int(tf["lookback_closes"])
            v = (close - close.shift(lb)).fillna(0.0)
            votes.append(np.sign(v))
        votes = pd.concat(votes, axis=1)
        bull_votes = (votes > 0).sum(axis=1)
        bear_votes = (votes < 0).sum(axis=1)
        regime = pd.Series(0, index=close.index, dtype=int)
        regime[bull_votes >= self.req] = 1
        regime[bear_votes >= self.req] = -1
        return regime
