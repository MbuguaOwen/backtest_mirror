from collections import deque

class EntryManager:
    def __init__(self, cfg):
        self.direction = cfg["entry"]["direction"]
        self.cooldown_bars = int(cfg["entry"]["cooldown_bars"])
        self._cooldown = 0

    def can_long(self): return self.direction in ("long","both")
    def can_short(self): return self.direction in ("short","both")

    def tick(self):
        if self._cooldown > 0:
            self._cooldown -= 1

    def arm_cooldown(self):
        self._cooldown = self.cooldown_bars

    def ready(self) -> bool:
        return self._cooldown == 0
