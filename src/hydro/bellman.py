import numpy as np
from scipy.interpolate import interp1d

from hydro.cost_function import HydroCostFunction
from hydro.reservoir import HydroReservoir
from base.bellman import Bellman
import constants


class HydroBellman(Bellman):
    """Bellman class for hydro. Inherits from Bellman.

    Computes and provides bellman values and penalties for each week.

    Attributes:
        penalty_factor (float): factor to modulate how important it is to respect guidelines.
    """
    def __init__(self, list_sce: np.ndarray, penalty_factor: float, cost_function: HydroCostFunction, reservoir: HydroReservoir):
        super().__init__(list_sce, cost_function, reservoir)
        self.penalty_factor = penalty_factor

    def get_penalty(self, week_idx: int, stock: float) -> float:
        """
        Returns a piecewise penalty function penalizing deviations outside the weekly lower and upper rule curves.
        Penalties grow linearly beyond ±1% of reservoir capacity from the rule curves.
        """

        assert isinstance(self._cost_function, HydroCostFunction)
        assert isinstance(self._reservoir.upper_guide, np.ndarray)

        max_cost = self._cost_function.max_cost(week_idx)
        lower = self._reservoir.lower_guide[week_idx]
        upper = self._reservoir.upper_guide[week_idx]
        cap = self._reservoir.capacity
        mid = 0.5 * (lower + upper)
        alpha = self._cost_function.alpha
        if week_idx == constants.RESULTS_SIZE - 1:
            res = 10 * max_cost * abs(stock - self._reservoir.final_level) / self._reservoir.capacity
        else:
            res = self.penalty_factor * self._cost_function.max_cost(week_idx) * ((stock - mid) / cap) ** alpha
        return res

    def bellman_function(self, week: int) -> interp1d:
        """
        Returns an interpolated Bellman value function for given week over reservoir stock levels.
        """
        if self._bellman_values is None:
            self._compute_bellman_values()
        assert isinstance(self._bellman_values, np.ndarray)
        stocks = np.linspace(0, self._reservoir.capacity, 100 // self._reservoir.step + 1)
        return interp1d(
            stocks,
            self._bellman_values[week],
            kind="linear",
            fill_value="extrapolate"
        )