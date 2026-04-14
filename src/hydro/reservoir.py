from dataclasses import dataclass, field
import numpy as np

from reservoir import Reservoir
import constants


@dataclass
class HydroReservoir(Reservoir):
    """Reservoir class for hydro. Inherits from Reservoir

    Attributes:
        weekly_max_turb (np.array): maximum amount of energy that can be produced for each week
        weekly_max_pump (np.array): maximum amount of energy that can be used to pump water for each week
        hourly_max_turb (np.array): maximum amount of energy that can be produced for each hour
        hourly_max_pump (np.array): maximum amount of energy that can be used to pump water for each hour
        turb_efficiency (float): coefficient for turbine efficiency
        pump_efficiency (float): coefficient for pumping efficience
    """
    weekly_max_turb: np.ndarray = field(default_factory=
                                        lambda: np.ones(shape=constants.RESULTS_SIZE, dtype=np.float64))
    weekly_max_pump: np.ndarray = field(default_factory=
                                        lambda: np.ones(shape=constants.RESULTS_SIZE, dtype=np.float64))
    hourly_max_turb: np.ndarray = field(default_factory=
                                        lambda: np.ones(shape=constants.NB_HOURS, dtype=np.float64)/(7*24))
    hourly_max_pump: np.ndarray = field(default_factory=
                                        lambda: np.ones(shape=constants.NB_HOURS, dtype=np.float64)/(7*24))
    turb_efficiency: float = 1
    pump_efficiency: float = 1
    