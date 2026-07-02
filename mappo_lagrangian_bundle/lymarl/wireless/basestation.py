import numpy as np
from typing import Tuple
from collections import deque


class BaseStation:
    """Base class representing a general base station."""

    def __init__(self,
                 bs_id: int,
                 position: Tuple[float, float],
                 tx_power_dbm: float = None,
                 frequency: float = None,
                 antenna_gain_tx: float = None,
                 antenna_gain_rx: float = None,
                 bandwidth: float = None,
                 shadowing_std_dev: float = None,
                 path_loss_exp: float = None,
                 reference_distance: float = None,
                 noise_figure_db: float = None,
                 beam_limit: int = np.inf,
                 config_dict: dict = None):
        if config_dict:
            self.bs_id = bs_id
            self.position = config_dict.get('position', position)
            self.tx_power_dbm = config_dict.get('tx_power_dbm', tx_power_dbm)
            self.frequency = config_dict.get('frequency_mhz', frequency)
            self.antenna_gain_tx = config_dict.get('antenna_gain_tx', antenna_gain_tx)
            self.antenna_gain_rx = config_dict.get('antenna_gain_rx', antenna_gain_rx)
            self.bandwidth = config_dict.get('bandwidth_mhz', bandwidth)
            self.shadowing_std_dev = config_dict.get('shadowing_std_dev', shadowing_std_dev)
            self.path_loss_exp = config_dict.get('path_loss_exp', path_loss_exp)
            self.reference_distance = config_dict.get('reference_distance', reference_distance)
            self.noise_figure_db = config_dict.get('noise_figure_db', noise_figure_db)
            self.beam_limit = config_dict.get('beam_limit', beam_limit)
        else:
            self.bs_id = bs_id
            self.position = position
            self.tx_power_dbm = tx_power_dbm
            self.frequency = frequency
            self.antenna_gain_tx = antenna_gain_tx
            self.antenna_gain_rx = antenna_gain_rx
            self.bandwidth = bandwidth
            self.shadowing_std_dev = shadowing_std_dev
            self.path_loss_exp = path_loss_exp
            self.reference_distance = reference_distance
            self.noise_figure_db = noise_figure_db
            self.beam_limit = beam_limit
        self.wavelength = 3e8 / self.frequency
        self.connected_users = []

    def distance_to(self, user_pos) -> float:
        return np.linalg.norm(np.array(user_pos) - np.array(self.position))

    def path_loss(self, distance):
        if self.reference_distance is None:
            raise ValueError("Reference distance must be set for path loss calculation.")
        fspl_db = 20 * np.log10(4 * np.pi * self.reference_distance / self.wavelength)
        pl_db = fspl_db + 10 * self.path_loss_exp * np.log10(distance / self.reference_distance)
        shadowing = np.random.normal(0, self.shadowing_std_dev) if self.shadowing_std_dev else 0
        return pl_db + shadowing

    def receive_power(self, distance) -> float:
        if self.tx_power_dbm is None:
            raise ValueError("Transmit power must be set for receive power calculation.")
        path_loss_db = self.path_loss(distance)
        return self.tx_power_dbm + self.antenna_gain_tx + self.antenna_gain_rx - path_loss_db

    def can_serve(self, user_pos):
        return True

    def reset(self):
        self.connected_users = []


class MacroBaseStation(BaseStation):
    """Macro base station with full area coverage."""

    def __init__(self, bs_id, position):
        super().__init__(bs_id=bs_id,
                         position=position,
                         tx_power_dbm=46,
                         frequency=2e9,
                         antenna_gain_tx=17,
                         antenna_gain_rx=0,
                         bandwidth=10e6,
                         shadowing_std_dev=np.sqrt(9),
                         path_loss_exp=3.76,
                         reference_distance=20.7,
                         noise_figure_db=5,
                         beam_limit=np.inf)

    def can_serve(self, user_pos):
        return True


class SmallCellBaseStation(BaseStation):
    """Small cell base station with limited coverage radius (default 35 m)."""

    def __init__(self, bs_id, position, beam_limit, coverage_radius: float = 35, tx_power_dbm: float = 20):
        # Why: Allow per-BS tx_power_dbm so we can introduce BS heterogeneity without subclassing.
        super().__init__(bs_id=bs_id,
                         position=position,
                         tx_power_dbm=tx_power_dbm,
                         frequency=28e9,
                         antenna_gain_tx=0,
                         antenna_gain_rx=0,
                         bandwidth=500e6,
                         shadowing_std_dev=np.sqrt(12),
                         path_loss_exp=2.5,
                         reference_distance=5,
                         noise_figure_db=0,
                         beam_limit=beam_limit)
        self.coverage_radius = coverage_radius

    def can_serve(self, user_pos):
        return True

    def path_loss(self, distance):
        return 128.1 + 37.6 * np.log10(distance / 1000)
