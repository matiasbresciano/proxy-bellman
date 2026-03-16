import numpy as np

from proxy import Proxy
from hydro.trajectories import HydroTrajectory
from hydro.bellman import HydroBellman
from hydro.cost_function import HydroCostFunction
from hydro.reservoir import HydroReservoir


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
