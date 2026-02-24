from dataclasses import dataclass, field
import numpy as np

from reservoir import Reservoir
import constants


@dataclass
class HydroReservoir(Reservoir):
    weekly_max_turb: np.ndarray = field(default_factory=
                                        lambda: np.zeros(shape=constants.RESULTS_SIZE, dtype=np.float64))
    weekly_max_pump: np.ndarray = field(default_factory=
                                        lambda: np.zeros(shape=constants.RESULTS_SIZE, dtype=np.float64))
    hourly_max_turb: np.ndarray = field(default_factory=
                                        lambda: np.zeros(shape=constants.NB_HOURS, dtype=np.float64))
    hourly_max_pump: np.ndarray = field(default_factory=
                                        lambda: np.zeros(shape=constants.NB_HOURS, dtype=np.float64))
    turb_efficiency: float = 1
    pump_efficiency: float = 1
    