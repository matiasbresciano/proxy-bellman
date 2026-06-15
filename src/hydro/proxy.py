import numpy as np
import typing

from base.proxy import Proxy, AntaresProxy
from hydro.trajectory import HydroTrajectory
from hydro.bellman import HydroBellman
from hydro.cost_function import HydroCostFunction
from hydro.reservoir import HydroReservoir
import constants
from hydro.study_modifier import StudyModifier


class HydroProxy(Proxy):
    """Proxy class for hydro.

    Manages computation for hydro trajectories and controls upon a given set of scenarii.
    """
    def __init__(self,
                 residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]],
                 reservoir: typing.List[HydroReservoir],
                 mc_years: np.ndarray,
                 ts_selection: np.ndarray | None = None,
                 turb_threshold: int = 25,
                 alpha: int = 2,
                 penalty_factor: float = 1) -> None:
        """Initialises the proxy

        Parameters:
            residual_load: residual_load of the different scenarios provided (hourly)
            reservoir: the reservoir used for the simulation
            mc_years (np.ndarray): list of scenarii for which we want to compute the trajectory.
            ts_selection (np.ndarray | None): list of scenarii to take into account for the computation of the bellman values.
            turb_threshold (int): number of values on which the cost function is computed (default is 25)
            alpha (int): parameter for the computation of the costs value and the turbine vs pumping ratio
            penalty_factor (float): factor to modulate how important it is to respect guidelines
        """
        super().__init__(residual_load, list(reservoir), mc_years, ts_selection)
        cost_function = HydroCostFunction(self._residual_load, reservoir[0], turb_threshold, alpha)
        self._cost_function.append(cost_function)
        bellman = HydroBellman(self.ts_selection, penalty_factor, cost_function, reservoir[0])
        self._bellman.append(bellman)
        trajectories = HydroTrajectory(self.mc_years, reservoir[0], cost_function, bellman)
        self._trajectory.append(trajectories)


class HydroAntaresProxy(AntaresProxy):
    """This class manages the computation of Bellman values and trajectory regarding a hydro reservoir
    using the scenarii of a given antares study."""
    def __init__(self, study_path: str,
                 area_name: str,
                 mc_years: np.ndarray,
                 sce_selection: np.ndarray | None = None,
                 turb_threshold: int = 25,
                 alpha: int = 2,
                 penalty_factor: float = 1):
        """Initialises the proxy using an antares study.

        Parameters:
            area_name (str): name of the area to consider.
            mc_years (np.ndarray): list of scenarii for which we want to compute the trajectory.
            sce_selection (np.ndarray | None): list of scenarii to take into account for the computation of the bellman values.
            turb_threshold (int): number of values on which the cost function is discretised (default is 25).
            alpha (int): parameter for the computation of the costs value and the turbine vs pumping ratio.
            penalty_factor (float): factor to modulate how important it is to respect guidelines.
        """
        super().__init__(study_path, area_name, mc_years, sce_selection)
        area = self.study.get_areas()[self.area]
        capacity = area.hydro.properties.reservoir_capacity
        lower_guide = area.hydro.get_reservoir()[0][7::7].values * capacity
        upper_guide = area.hydro.get_reservoir()[2][7::7].values * capacity
        initial_level = (area.hydro.get_reservoir()[0][0] + area.hydro.get_reservoir()[2][0]) / 2 * capacity
        final_level = initial_level
        daily_inflow = self._add_mc_years(area.hydro.get_mod_series()[:constants.NB_DAYS].values)
        hourly_inflow = np.repeat(daily_inflow/constants.NB_HOURS_IN_DAY, constants.NB_HOURS_IN_DAY, axis=0)
        max_turb = area.hydro.get_maxpower()[0][:constants.NB_DAYS].values
        max_pump = area.hydro.get_maxpower()[2][:constants.NB_DAYS].values
        weekly_turb = (max_turb * constants.NB_HOURS_IN_DAY).reshape(
            (constants.RESULTS_SIZE, constants.RESULTS_INTERVAL_DAYS)
            ).sum(axis=1)
        weekly_pump = (max_pump * constants.NB_HOURS_IN_DAY).reshape(
            (constants.RESULTS_SIZE, constants.RESULTS_INTERVAL_DAYS)
            ).sum(axis=1)
        hourly_turb = np.repeat(max_turb, constants.NB_HOURS_IN_DAY)
        hourly_pump = np.repeat(max_pump, constants.NB_HOURS_IN_DAY)
        turb_eff = 1
        pump_eff = area.hydro.properties.pumping_efficiency
        self._reservoir = HydroReservoir(capacity=capacity,
                                   lower_guide=lower_guide,
                                   upper_guide=upper_guide,
                                   initial_level=initial_level,
                                   final_level=final_level,
                                   hourly_inflow=hourly_inflow,
                                   weekly_max_turb=weekly_turb,
                                   weekly_max_pump=weekly_pump,
                                   hourly_max_turb=hourly_turb,
                                   hourly_max_pump=hourly_pump,
                                   turb_efficiency=turb_eff,
                                   pump_efficiency=pump_eff,
                                   step=2)

        # weights
        for alloc in area.hydro.allocation:
            load = self._area_loads[alloc.area_id].astype(dtype=np.float64) * alloc.coefficient
            self._residual_load = self._residual_load + load

        self._proxy = HydroProxy(self._residual_load, [self._reservoir], self.mc_years, self.sce_selection, turb_threshold, alpha, penalty_factor)

    def apply_to_study(self) -> None:
        """This method applies the computed trajectories to the study."""
        nb_sce = self._residual_load.shape[1]
        traj = self._proxy._trajectory[0]
        assert isinstance(traj, HydroTrajectory)
        study_modifier = StudyModifier(nb_sce, self._reservoir, traj, self.study_path, self.area)
        study_modifier.apply_all()

    def undo_study(self) -> None:
        """Removes the modification done by apply_to_study."""
        nb_sce = self._residual_load.shape[1]
        traj = self._proxy._trajectory[0]
        assert isinstance(traj, HydroTrajectory)
        study_modifier = StudyModifier(nb_sce, self._reservoir, traj, self.study_path, self.area)
        study_modifier.undo_all()

