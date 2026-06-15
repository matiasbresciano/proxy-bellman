"""
Base for the Proxy classes
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import typing
import antares.craft as ac
from antares.craft.model.study import Study
import pandas as pd
import os

from base.cost_function import CostFunction
from base.reservoir import Reservoir
from base.bellman import Bellman
from base.trajectory import Trajectory
import constants


class Proxy(ABC):
    """This abstract class is a model for Bellman values and trajectory computation classes.

    Attributes:
        _residual_load (np.ndarray): residual_load of the different scenarios provided (hourly).
        _day_of_first_data (int): day of the year corresponding to the first data (0 (january 1st) to 364 (december 31st)).
        _week_day_of_first_data (int): day of the week of first day of data (0 for monday, 6 for sunday).
        mc_years (np.ndarray(int)): list of years for which to compute trajectories.
        ts_selection (np.ndarray(int)): list of years to take into account to compute bellman values.
        _reservoir (list(Reservoir)): Reservoirs describing the different stocks.
        _cost_function (list(GainFunction): gain function calculators.
        _bellman (list(Bellman)): Bellman values calculators.
        _trajectory (list(Trajectory)): The trajectories calculators.
    """

    def __init__(self,
                 residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]],
                 reservoir: typing.List[Reservoir],
                 mc_years: np.ndarray | None = None,
                 ts_selection: np.ndarray | None = None,
                 day_of_first_data: int = 0,
                 week_day_of_first_data: int = 0) -> None:
        """Initialises the proxy

        Parameters:
            residual_load: residual_load of the different scenarios provided (hourly).
            reservoir: the reservoir used for the simulation.
            mc_years (np.ndarray(int)): list of years for which to compute trajectories.
            ts_selection (np.ndarray(int)): list of years to take into account to compute bellman values.
            day_of_first_data: day of the year corresponding to the first data (0 (january 1st) to 364 (december 31st)).
            week_day_of_first_data: day of the week of the first data (0 (monday) to 6 (sunday)).
        """
        self._residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]] = residual_load
        self._day_of_first_data: int = day_of_first_data
        self._week_day_of_first_data: int = week_day_of_first_data
        self._reservoir: typing.List[Reservoir] = reservoir
        if mc_years is not None:
            self.mc_years: np.ndarray = mc_years
        else:
            self.mc_years = np.arange(residual_load.shape[1])
        if ts_selection is not None:
            self.ts_selection: np.ndarray = ts_selection
        else:
            self.ts_selection = self.mc_years
        self._cost_function: typing.List[CostFunction] = []
        self._bellman: typing.List[Bellman] = []
        self._trajectory: typing.List[Trajectory] = []

    def get_trajectories(self) -> typing.List[np.ndarray]:
        """
        Returns the computed trajectories. The array is indexed as traj[scenario_index, week_index].
        """
        res: list[np.ndarray] = []
        for t in self._trajectory:
            res.append(t.get_trajectories())
        return res

    def get_controls(self) -> typing.List[np.ndarray]:
        """
        Returns the computed controls. The array is indexed as traj[scenario_index, week_index].
        """
        res: list[np.ndarray] = []
        for t in self._trajectory:
            res.append(t.get_controls())
        return res

    def get_usage_values(self) -> typing.List[np.ndarray]:
        """
        Returns the usage values. The array is indexed as traj[scenario_index, week_index].
        """
        res: list[np.ndarray] = []
        for b in self._bellman:
            res.append(b.get_usage_values())
        return res

    def get_bellman_values(self) -> typing.List[np.ndarray]:
        """
        Returns the computed Bellman values. The array is indexed as traj[week_index, level_index].
        """
        res: list[np.ndarray] = []
        for b in self._bellman:
            res.append(b.get_bellman_values())
        return res


class AntaresProxy(ABC):
    """This abstract class is a model for Bellman values and trajectory computation classes from an antares study.

    Attributes:
        study_path (str): path to the antares study.
        study (Study): antares study.
        area (str): name of the area to consider.
        mc_years (int): number of MC years.
        sce_selection (np.ndarray): list of MC years to take into account (ignores nb_sce if present).
        _area_loads (dict[str, np.ndarray]): computed load of each area in the study.
        _residual_load (np.ndarray[tuple[int, int], np.dtype[np.float64]]): residual load to consider for the
            computations.
        _proxy (Proxy): computation unit.
    """

    def __init__(self, study_path: str, area: str, mc_years: np.ndarray, sce_selection: np.ndarray | None) -> None:
        self.study_path: str = study_path
        self.area: str = area
        self.study: Study = ac.read_study_local(Path(study_path))
        self.mc_years: np.ndarray = mc_years
        self.sce_selection: np.ndarray = mc_years
        if sce_selection is not None:
            self.sce_selection = sce_selection
        self._area_loads: dict[str, np.ndarray] = dict()
        self._compute_area_residual_loads()
        self._residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]] \
            = np.zeros(shape=self._area_loads[area].shape, dtype=np.float64)
        self._proxy: Proxy = Proxy(self._residual_load, [])

    def _compute_area_residual_loads(self) -> None:
        """
        Computes the residual load of each area of the study, taking into account the initial load and the energy
        provided.
        """
        for ar_name, ar_value in self.study.get_areas().items():
            load = self._add_mc_years(ar_value.get_load_matrix().values)
            renewables = np.zeros(shape=load.shape, dtype=np.float64)
            for ren in ar_value.get_renewables().values():
                renewables += self._add_mc_years(ren.get_timeseries().values * ren.properties.nominal_capacity)
            solar = self._add_mc_years(ar_value.get_solar_matrix().values)
            wind = self._add_mc_years(ar_value.get_wind_matrix().values)
            ror = self._add_mc_years(ar_value.hydro.get_ror_series().values[:load.shape[0], :])
            misc = self._add_mc_years(ar_value.get_misc_gen_matrix().values.sum(axis=1)[:, np.newaxis])
            self._area_loads[ar_name] = load - ror - misc - renewables - solar - wind

    def _add_mc_years(self, array: np.ndarray) -> np.ndarray:
        """
        Loop the array if it does not contain enough time series, troncates it if it contains too many.
        Selects the right time series if sce_selection is present.
        """
        nb_sce = np.max(self.sce_selection) + 1
        nb_sce = max(nb_sce, np.max(self.mc_years) + 1)
        while array.shape[1] < nb_sce:
            array = np.concatenate((array, array), axis=1)
        if array.shape[1] > nb_sce:
            array = array[:, :nb_sce]
        return np.asarray(array, dtype=np.float64)

    def get_trajectories(self) -> list[np.ndarray]:
        """
        Returns the computed trajectories. The array is indexed as traj[scenario_index, week_index].
        """
        return self._proxy.get_trajectories()

    def get_controls(self) -> list[np.ndarray]:
        """
        Returns the computed controls. The array is indexed as traj[scenario_index, week_index].
        """
        return self._proxy.get_controls()

    def get_bellman_values(self) -> list[np.ndarray]:
        """
        Returns the computed Bellman values. The array is indexed as traj[week_index, level_index].
        """
        return self._proxy.get_bellman_values()

    def save_residual_loads(self) -> None:
        """
        Saves the computed residual load for each scenario in the user folder.
        """
        sb_dir = os.path.join(self.study_path, "user")
        os.makedirs(sb_dir, exist_ok=True)
        np.savetxt(os.path.join(sb_dir, "residual_loads.txt"), self._residual_load)

    @staticmethod
    def _int_from_antares_weekday(weekday: ac.WeekDay) -> int:
        """
        Parses antares weekdays to get the corresponding int.
        """
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
        """
        Parses antares months to get the corresponding int.
        """
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
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
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
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        output_path = os.path.join(export_dir, filename)
        df.to_csv(output_path, index=False)
