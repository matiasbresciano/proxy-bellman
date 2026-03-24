from dataclasses import dataclass, field
import numpy as np

from reservoir import Reservoir
import constants


@dataclass
class HydroReservoir(Reservoir):
    """Reservoir class for tempo. Inherits from Reservoir

    Attributes:
        excluded_week_days: days of the week when a tempo day cannot be used
        first_day: first day of the year when a tempo day can be used (included) (1st of november for red)
        last_day: last day of the year when a tempo day can be used (included) (31st of marsh for red)
    """
    excluded_week_days: np.ndarray = field(default_factory=
                                           lambda: np.asarray([6], dtype=np.int16))
    first_day: int = 304  # november the 1st
    last_day: int = 89  # marsh the 31st

