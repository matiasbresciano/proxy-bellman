from abc import ABC, abstractmethod
import numpy as np
import typing

from gain_function import GainFunction
from reservoir import Reservoir
from bellman import Bellman
from trajectory import Trajectory

"""
Base for the Proxy classes
"""


class Proxy(ABC):
    """This abstract class is a model for Bellman values computation classes

    Attributes:
        _residual_load (np.ndarray): residual_load of the different scenarios provided (hourly)
        _reservoir (list(Reservoir)): Reservoirs describing the different stocks
        _gain_function (list(GainFunction): gain function calculators
        _bellman (list(Bellman)): Bellman values calculators
        _trajectory (list(Trajectory)): The trajectories calculators
    """
    _residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    _day_of_first_data: int
    _week_day_of_first_data: int
    _reservoir: typing.List[Reservoir]
    _gain_function: typing.List[GainFunction]
    _bellman: typing.List[Bellman]
    _trajectory: typing.List[Trajectory]

    def __init__(self,
                 residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]],
                 day_of_first_data: int = 0,
                 week_day_of_first_data: int = 0) -> None:
        """Initialises the proxy

        Parameters:
            residual_load: residual_load of the different scenarios provided (hourly)
            day_of_first_data: day of the year corresponding to the first data (0 (january 1st) to 364 (december 31st))
            week_day_of_first_data: day of the week of the first data (0 (monday) to 6 (sunday))
        """
        self._residual_load = residual_load
        self._day_of_first_data = day_of_first_data
        self._week_day_of_first_data = week_day_of_first_data

    def get_trajectories(self) -> typing.List[np.ndarray]:
        res: list[np.ndarray] = []
        for t in self._trajectory:
            res += t.get_trajectories()
        return res

    def get_controls(self) -> typing.List[np.ndarray]:
        res: list[np.ndarray] = []
        for t in self._trajectory:
            res += t.get_controls()
        return res

    def get_usage_values(self) -> typing.List[np.ndarray]:
        res: list[np.ndarray] = []
        for b in self._bellman:
            res += b.get_usage_values()
        return res

    def get_bellman_values(self) -> typing.List[np.ndarray]:
        res = []
        for b in self._bellman:
            res += b.get_bellman_values()
        return res
