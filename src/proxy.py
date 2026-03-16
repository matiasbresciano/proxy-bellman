from abc import ABC, abstractmethod
import numpy as np
import typing
import antares.craft as ac

from cost_function import CostFunction
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
        _cost_function (list(GainFunction): gain function calculators
        _bellman (list(Bellman)): Bellman values calculators
        _trajectory (list(Trajectory)): The trajectories calculators
    """
    _residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    _day_of_first_data: int
    _week_day_of_first_data: int
    _reservoir: typing.List[Reservoir]
    _cost_function: typing.List[CostFunction]
    _bellman: typing.List[Bellman]
    _trajectory: typing.List[Trajectory]

    def __init__(self,
                 residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]],
                 reservoir: Reservoir,
                 day_of_first_data: int = 0,
                 week_day_of_first_data: int = 0) -> None:
        """Initialises the proxy

        Parameters:
            residual_load: residual_load of the different scenarios provided (hourly)
            reservoir: the reservoir used for the simulation
            day_of_first_data: day of the year corresponding to the first data (0 (january 1st) to 364 (december 31st))
            week_day_of_first_data: day of the week of the first data (0 (monday) to 6 (sunday))
        """
        self._residual_load = residual_load
        self._day_of_first_data = day_of_first_data
        self._week_day_of_first_data = week_day_of_first_data
        self._reservoir = [reservoir]
        self._cost_function = []
        self._bellman = []
        self._trajectory = []

    def get_trajectories(self) -> typing.List[np.ndarray]:
        res: list[np.ndarray] = []
        for t in self._trajectory:
            res.append(t.get_trajectories())
        return res

    def get_controls(self) -> typing.List[np.ndarray]:
        res: list[np.ndarray] = []
        for t in self._trajectory:
            res.append(t.get_controls())
        return res

    def get_usage_values(self) -> typing.List[np.ndarray]:
        res: list[np.ndarray] = []
        for b in self._bellman:
            res.append(b.get_usage_values())
        return res

    def get_bellman_values(self) -> typing.List[np.ndarray]:
        res: list[np.ndarray] = []
        for b in self._bellman:
            res.append(b.get_bellman_values())
        return res


class AntaresProxy(ABC):
    """This abstract class is a model for Bellman values computation classes from an antares study

    Attributes:
        study_path (str): path to the antares study
        area (str): name of the area to consider
        _proxy (Proxy): computation unit
    """
    study_path: str
    area: str
    _proxy: Proxy
    _residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    def __init__(self, study_path: str, area: str, mc_years: int, sce_selection: list[int] | None) -> None:
        self.study_path = study_path
        self.area = area
        self._residual_load = ac.read_study_local(study_path).get_areas()
