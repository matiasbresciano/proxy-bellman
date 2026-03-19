import numpy as np

from proxy import Proxy, AntaresProxy
from hydro.trajectories import HydroTrajectory
from hydro.bellman import HydroBellman
from hydro.cost_function import HydroCostFunction
from hydro.reservoir import HydroReservoir
import constants


class HydroProxy(Proxy):
    """Proxy class for hydro. Inherits from Hydro

    Manages computation for hydro trajectories and controls upon a given set of scenario
    """
    def __init__(self,
                 residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]],
                 reservoir: HydroReservoir,
                 turb_threshold: int = 25,
                 alpha: int = 2,
                 penalty_factor: int = 1) -> None:
        """Initialises the proxy

        Parameters:
            residual_load: residual_load of the different scenarios provided (hourly)
            reservoir: the reservoir used for the simulation
            turb_threshold (int): number of values on which the cost function is computed (default is 25)
            alpha (int): parameter for the computation of the costs value and the turbine vs pumping ratio
            penalty_factor (float): factor to modulate how important it is to respect guidelines
        """
        super().__init__(residual_load, reservoir)
        nb_sce = residual_load.shape[1]
        cost_function = HydroCostFunction(self._residual_load, reservoir, turb_threshold, alpha)
        self._cost_function.append(cost_function)
        bellman = HydroBellman(nb_sce, penalty_factor, cost_function, reservoir)
        self._bellman.append(bellman)
        trajectories = HydroTrajectory(nb_sce, reservoir, cost_function, bellman)
        self._trajectory.append(trajectories)


class HydroAntaresProxy(AntaresProxy):
    def __init__(self, study_path: str,
                 area_name: str,
                 mc_years: int,
                 sce_selection: list[int] | None = None,
                 turb_threshold: int = 25,
                 alpha: int = 2,
                 penalty_factor: int = 1):
        super().__init__(study_path, area_name, mc_years, sce_selection)
        area = self.study.get_areas()[self.area]
        capacity = area.hydro.properties.reservoir_capacity
        lower_guide = area.hydro.get_reservoir()[0][7::7].values * capacity
        upper_guide = area.hydro.get_reservoir()[1][7::7].values * capacity
        initial_level = (lower_guide[0] + upper_guide[0]) / 2
        # final_level = (lower_guide[-1] + upper_guide[-1]) / 2  # TODO LRI: vérifier qu'on garde ça
        final_level = initial_level
        daily_inflow = area.hydro.get_mod_series()[:constants.NB_DAYS]
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
        # turb_eff = 1
        pump_eff = area.hydro.properties.pumping_efficiency
        reservoir = HydroReservoir(capacity=capacity,
                                   lower_guide=lower_guide,
                                   upper_guide=upper_guide,
                                   initial_level=initial_level,
                                   final_level=final_level,
                                   hourly_inflow=hourly_inflow,
                                   weekly_max_turb=weekly_turb,
                                   weekly_max_pump=weekly_pump,
                                   hourly_max_turb=hourly_turb,
                                   hourly_max_pump=hourly_pump,
                                   # turb_efficiency=turb_eff,
                                   pump_efficiency=pump_eff,
                                   step=2)

        # weights
        for alloc in area.hydro.allocation:
            load = self._area_loads[alloc.area_id] * alloc.coefficient
            self._residual_load = self._residual_load + load

        self._proxy = HydroProxy(self._residual_load, reservoir, turb_threshold, alpha, penalty_factor)
