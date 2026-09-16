"""
Base for the Bellman values computation classes
"""

from abc import ABC, abstractmethod
import numpy as np
import typing
import math

from base.cost_function import CostFunction
from base.reservoir import Reservoir
import constants


class Bellman(ABC):
    """This abstract class is a model for Bellman values computation classes.

    Attributes:
        _list_sce (np.ndarray): indexes of the scenarii to consider
        _reservoir (Reservoir): Reservoir describing the stock
        _cost_function (CostFunction): gain function to use for computing bellman values
        _bellman_values (np.ndarray): the value associated to each possible stock level for each week
        _usage_value (np.ndarray): the usage value associated to each week and each possible stock level
    """

    def __init__(self, list_sce: np.ndarray, cost_function: CostFunction, reservoir: Reservoir) -> None:
        self._list_sce: np.ndarray = list_sce
        self._cost_function: CostFunction = cost_function
        self._reservoir: Reservoir = reservoir
        self._bellman_values: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None
        self._usage_value: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None

    def get_bellman_value(self, week: int, stock: float) -> float:
        """Returns the bellman value associated to a week and stock"""
        if self._bellman_values is None:
            self._compute_bellman_values()
        assert isinstance(self._bellman_values, np.ndarray)
        assert isinstance(self._reservoir.possible_control_values, np.ndarray)
        stock_ratio = np.linspace(0, self._reservoir.capacity, len(self._reservoir.possible_control_values))
        res = float(np.interp(stock, stock_ratio, self._bellman_values[week]))
        return res

    def iterate_over_controls_vec(self, controls: np.ndarray, next_stock: np.ndarray,
                                  week_ind: int, sce_ind: int, exact_ctrls: bool = True)\
            -> tuple[float, float, float]:
        """
        Computes the best value over the different provided controls

        Parameters:
             controls (np.ndarray): different controls to test
             next_stock (np.ndarray): stock values corresponding to the controls
             week_ind (int): considered week
             sce_ind (int): considered scenario
             exact_ctrls (bool): controls correspond to the exact points in _cost_function, if false,
                needs interpolation

        Returns:
            (best value, corresponding next stock, corresponding control)
        """
        cost = np.asarray([self._cost_function.get_cost(week_ind, sce_ind, ctrl) for ctrl in controls])
        penalty = np.asarray([self.get_penalty(week_ind, stock) for stock in next_stock])
        bellman_value = np.asarray([self.get_bellman_value(week_ind, stock) for stock in next_stock])
        total_value = cost + penalty + bellman_value
        if week_ind == 51:
            total_value = cost + bellman_value
        j = int(np.argmin(total_value))
        return float(total_value[j]), float(next_stock[j]), float(controls[j])

    def iterate_over_stock_levels_vec(
            self,
            best_value: tuple[float, float | None, float | None],
            current_stock_with_inflow: float,
            week_ind: int,
            sce_ind: int,
            max_control: float
    ) -> tuple[float, float | None, float | None]:
        """
        Enumerates next stock levels on the 0..100% grid,
        computes implied control, filters infeasible controls, evaluates total value,
        and keeps the best.

        Parameters:
             best_value (np.ndarray): previously computed best value (over controls), corresponding next stock,
                corresponding control
             current_stock_with_inflow (float): stock values corresponding to the controls
             week_ind (int): considered week
             sce_ind (int): considered scenario
             max_control: max_control possible

        Returns:
            best value, corresponding next stock, corresponding control
        """
        assert isinstance(self._reservoir.possible_control_values, np.ndarray)
        next_stock_grid = self._reservoir.possible_control_values

        controls = current_stock_with_inflow - next_stock_grid
        feasible = self._reservoir.feasibility(controls, week_ind, max_control)

        if not np.any(feasible):
            return best_value

        ns = next_stock_grid[feasible]
        ctrl = controls[feasible]

        penalty = [self.get_penalty(week_ind, stock) for stock in ns]
        cost = [self._cost_function.get_cost(week_ind, sce_ind, c) for c in ctrl]

        bellman_values = np.asarray([self.get_bellman_value(week_ind, stock) for stock in ns])
        total_value = cost + bellman_values + penalty

        j = int(np.argmin(total_value))
        cand_value = float(total_value[j])

        if cand_value < best_value[0]:
            best_value = (cand_value, float(ns[j]),  float(ctrl[j]))

        return best_value

    def _compute_bellman_values(self) -> None:
        """
        Computes Bellman values at the end of each week by backward induction over weeks and scenarios.
        Applies penalties and selects optimal controls to minimize cost-to-go.
        """
        assert isinstance(self._reservoir.possible_control_values, np.ndarray)
        self._bellman_values = np.zeros(shape=(constants.RESULTS_SIZE, len(self._reservoir.possible_control_values)), dtype=np.float64)

        self._bellman_values[constants.RESULTS_SIZE - 1] = np.array([
            self.get_penalty(constants.RESULTS_SIZE - 1, c)
            for c in self._reservoir.possible_control_values
        ])

        for week_ind in reversed(range(constants.RESULTS_SIZE - 1)):

            for idx, current_stock in enumerate(self._reservoir.possible_control_values):
                bv_sce = np.zeros(len(self._list_sce))
                for i, sce_ind in enumerate(self._list_sce):
                    nb_inflow_sce = self._reservoir.hourly_inflow.shape[1]
                    weekly_inflow = self._reservoir.hourly_inflow[
                        (week_ind + 1) * constants.RESULTS_INTERVAL_HOURS:
                        (week_ind + 2) * constants.RESULTS_INTERVAL_HOURS, sce_ind % nb_inflow_sce
                                    ].sum(axis=0)
                    controls = self._cost_function.get_controls(week_ind + 1, sce_ind)
                    next_stock = current_stock + weekly_inflow - controls

                    best_value = self.iterate_over_controls_vec(
                        controls=controls,
                        next_stock=next_stock,
                        week_ind=week_ind + 1,
                        sce_ind=sce_ind
                    )

                    final_best_value, _, _ = self.iterate_over_stock_levels_vec(
                        best_value=best_value,
                        current_stock_with_inflow=current_stock + weekly_inflow,
                        week_ind=week_ind + 1,
                        sce_ind=sce_ind,
                        max_control=controls[-1])

                    bv_sce[i] = final_best_value

                self._bellman_values[week_ind, idx] = np.mean(bv_sce)


    def _compute_usage_values(self) -> None:
        assert isinstance(self._reservoir.possible_control_values, np.ndarray)
        self._usage_value = np.zeros(
            shape=(constants.RESULTS_SIZE, len(self._reservoir.possible_control_values)),
            dtype=np.float64)
        assert isinstance(self._bellman_values, np.ndarray)  # to avoid typing errors
        for w in range(constants.RESULTS_SIZE):
            for c in range(math.ceil(self._reservoir.capacity)):
                self._usage_value[w, c - 1] = self._bellman_values[w, c] - self._bellman_values[w, c - 1]

    def get_bellman_values(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Returns all bellman values. Array is indexed as [week_index, level]."""
        if self._bellman_values is None:
            self._compute_bellman_values()
        assert isinstance(self._bellman_values, np.ndarray)
        return self._bellman_values

    @abstractmethod
    def get_penalty(self, week: int, stock: float|int) -> float:
        """Returns the computed penalty for given week and stock"""
        pass

    def get_usage_values(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Returns all usage values. Array is indexed as [week_index, level_index]."""
        if self._usage_value is None:
            self._compute_usage_values()
        assert isinstance(self._usage_value, np.ndarray)
        return self._usage_value
