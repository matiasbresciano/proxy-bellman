from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import typing
import antares.craft as ac
from antares.craft.model.study import Study
import pandas as pd
import os

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
                 reservoir: typing.List[Reservoir],
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
        self._reservoir = reservoir
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
    study: Study
    _proxy: Proxy
    _area_loads: dict[str, np.ndarray]
    _residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    def __init__(self, study_path: str, area: str, mc_years: int, sce_selection: list[int] | None) -> None:
        self.study_path = study_path
        self.area = area
        self.study = ac.read_study_local(Path(study_path))
        # TODO LRI : ajouter gestion de mc_years, sce_selection
        self._area_loads = dict()
        self._compute_area_residual_loads()
        self._residual_load = np.zeros(shape=self._area_loads[area].shape, dtype=np.float64)

    def _compute_area_residual_loads(self) -> None:
        for ar_name, ar_value in self.study.get_areas().items():
            load = ar_value.get_load_matrix().values
            renewables = np.zeros(shape=load.shape, dtype=np.float64)
            for ren in ar_value.get_renewables().values():
                renewables += ren.get_timeseries().values * ren.properties.nominal_capacity
            ror = ar_value.hydro.get_ror_series().values[:load.shape[0], :load.shape[1]]
            misc = ar_value.get_misc_gen_matrix().values.sum(axis=1)
            self._area_loads[ar_name] = load - ror - misc[:, np.newaxis] - renewables

    def get_trajectories(self) -> list[np.ndarray]:
        return self._proxy.get_trajectories()

    def get_controls(self) -> list[np.ndarray]:
        return self._proxy.get_controls()

    def get_bellman_values(self) -> list[np.ndarray]:
        return self._proxy.get_bellman_values()

    @staticmethod
    def _int_from_antares_weekday(weekday: ac.WeekDay) -> int:
        res = 0
        match weekday:
            case ac.WeekDay.TUESDAY:
                res = 1
            case ac.WeekDay.WEDNESDAY:
                res = 2
            case ac.WeekDay.THURSDAY:
                res = 3
            case ac.WeekDay.FRIDAY:
                res = 4
            case ac.WeekDay.SATURDAY:
                res = 5
            case ac.WeekDay.SUNDAY:
                res = 6
        return res

    @staticmethod
    def _int_from_antares_month(month: ac.Month) -> int:
        res = 0
        match month:
            case ac.Month.FEBRUARY:
                res = 1
            case ac.Month.MARCH:
                res = 2
            case ac.Month.APRIL:
                res = 3
            case ac.Month.MAY:
                res = 4
            case ac.Month.JUNE:
                res = 5
            case ac.Month.JULY:
                res = 6
            case ac.Month.AUGUST:
                res = 7
            case ac.Month.SEPTEMBER:
                res = 8
            case ac.Month.OCTOBER:
                res = 9
            case ac.Month.NOVEMBER:
                res = 10
            case ac.Month.DECEMBER:
                res = 11
        return res

    def export_controls(self, export_dir: str, filename: str = "controls.csv") -> None:
        """
        Export optimal control trajectories
        for all scenarios and weeks to a CSV file.
        """
        controls = self._proxy.get_controls()[0]
        data = []
        for sce_ind, sce in enumerate(controls):
            for week_ind, val in enumerate(sce):
                u = val
                data.append({
                    "area": self.area,
                    "u": u,
                    "week": week_ind + 1,
                    "mcYear": sce_ind + 1
                })

        df = pd.DataFrame(data)
        output_path = os.path.join(export_dir, filename)
        df.to_csv(output_path, index=False)

    def export_trajectories(self, export_dir: str, filename: str = "trajectories.csv") -> None:
        """
        Export optimal stock trajectories for all scenarios and weeks
        to a CSV file.
        """
        trajectories = self._proxy.get_trajectories()[0]
        data = []

        for sce_ind, sce in enumerate(trajectories):
            for week_ind, val in enumerate(sce):
                hlevel = val
                data.append({
                    "area": self.area,
                    "hlevel": hlevel,
                    "week": week_ind + 1,
                    "mcYear": sce_ind + 1,
                })
        df = pd.DataFrame(data)
        output_path = os.path.join(export_dir, filename)
        df.to_csv(output_path, index=False)
