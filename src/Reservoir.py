from dataclasses import dataclass, field
import numpy as np

"""
Base modelisation of the stock on witch a trajectory must be computed
"""


# TODO LRI : changer 52 par ... TRAJECTORY_SIZE ? GUIDE_SIZE ?
@dataclass
class Reservoir:
    """This class is the base modelisation of the stock on witch a trajectory must be computed

    This class contains the details of the stock, such as its capacity, its potential inflows, and how and
    when it can be used.

    Attributes:
        capacity (int|float): the maximum amount in stock
        lower_guide (np.ndarray): the lower guiding curve, weekly values with default as 0
        upper_guide (np.ndarray): the upper guiding curve, weekly values with default as capacity
        initial_level (int|float): the initial amount in stock
        final_level (int|float): the intended final amount in stock
        hourly_inflow (np.ndarray): the added amount for each hour
    """
    capacity: int | float = 100
    lower_guide: np.ndarray = field(default_factory=lambda: np.zeros(shape=52, dtype=np.float64))
    upper_guide: np.ndarray = None
    initial_level: int | float = 0
    final_level: int | float = 0
    hourly_inflow: np.ndarray = field(default_factory=lambda: np.zeros(shape=8760, dtype=np.float64))

    def __post_init__(self):
        if self.upper_guide is None:
            self.upper_guide = self.capacity * np.ones(shape=52, dtype=np.float64)

