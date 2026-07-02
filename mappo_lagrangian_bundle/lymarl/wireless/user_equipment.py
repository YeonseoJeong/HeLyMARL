import numpy as np
from typing import Tuple


class UserEquipment:
    """User Equipment (UE) with position and per-BS SNR tracking."""

    def __init__(self, ue_id, position):
        self.ue_id = ue_id
        self.position = np.array(position)
        self.snr_list = {}

    def add_snr(self, bs_id, snr):
        self.snr_list[bs_id] = snr

    def best_bs(self):
        if not self.snr_list:
            return None
        candidates = {k: v for k, v in self.snr_list.items() if k != 0}
        return max(candidates, key=candidates.get) if candidates else None

    def get_snr(self, bs_id):
        return self.snr_list.get(bs_id, -np.inf)

    def __repr__(self):
        return f"UE#{self.ue_id} | [x, y] = {self.position} | Best BS: {self.best_bs()}"