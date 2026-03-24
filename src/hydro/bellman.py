import numpy as np
from scipy.interpolate import interp1d

from hydro.cost_function import HydroCostFunction
from hydro.reservoir import HydroReservoir
from bellman import Bellman
import constants


class HydroBellman(Bellman):
    """Bellman class for hydro. Inherits from Bellman

    Computes and provides bellman values and penalties for each week

    Attributes:
        penalty_factor (float): factor to modulate how important it is to respect guidelines
    """
    penalty_factor: float

    def __init__(self, nb_sce: int, penalty_factor: float, cost_function: HydroCostFunction, reservoir: HydroReservoir):
        super().__init__(nb_sce, cost_function, reservoir)
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

    def get_bellman_value(self, week: int, stock: float) -> float:
        if self._bellman_values is None:
            self._compute_bellman_values()
        assert isinstance(self._bellman_values, np.ndarray)
        stock_ratio = np.linspace(0, self._reservoir.capacity, 100 // self._reservoir.step + 1)
        res = np.interp(stock, stock_ratio, self._bellman_values[week])
        return res

    def iterate_over_controls_vec(self, controls: np.ndarray, next_stock: np.ndarray,
                                  week_ind: int, sce_ind: int, exact_ctrls: bool = True)\
            -> tuple[float, float, float]:
        """
        Computes the best value over the different provided controls

        Parameters:
             controls (np.ndarray): different controls to test
             next_stock (np.ndarray): stock values corresponding to the controls
             bellman_fn (interp1d): bellman fonction to interpolate bellman value
             week_ind (int): considered week
             sce_ind (int): considered scenario
             exact_ctrls (bool): controls correspond to the exact points in _cost_function, if false,
                needs interpolation

        Returns:
            best value, corresponding next stock, corresponding control
        """
        assert isinstance(self._cost_function, HydroCostFunction)
        cost = self._cost_function.get_exact_costs(week_ind, sce_ind)
        if not exact_ctrls:
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
        capacity = float(self._reservoir.capacity)

        next_stock_grid = (np.arange(0, 101, self._reservoir.step, dtype=float) / 100.0) * capacity

        controls = current_stock_with_inflow - next_stock_grid
        assert isinstance(self._reservoir, HydroReservoir)
        max_week_pump = self._reservoir.weekly_max_pump[week_ind]
        max_week_turb = self._reservoir.weekly_max_turb[week_ind]
        feasible = (
                (controls >= -max_week_pump * self._reservoir.pump_efficiency) &
                (controls <= max_week_turb * self._reservoir.turb_efficiency) &
                (controls <= max_control)
        )

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
        Computes Bellman values at end of each week by backward induction over weeks and scenarios.
        Applies penalties and selects optimal controls to minimize cost-to-go.
        """
        self._bellman_values = np.zeros(shape=(constants.RESULTS_SIZE, 100//self._reservoir.step + 1), dtype=np.float64)
        assert isinstance(self._reservoir, HydroReservoir)

        self._bellman_values[constants.RESULTS_SIZE - 1] = np.array([
            self.get_penalty(constants.RESULTS_SIZE - 1, c/100 * self._reservoir.capacity)
            for c in range(0, 101, self._reservoir.step)
        ])

        for week_ind in reversed(range(constants.RESULTS_SIZE - 1)):

            for c in range(0, 101, self._reservoir.step):
                current_stock = (c / 100) * self._reservoir.capacity
                bv_sce = np.zeros(self._nb_sce)
                for i in range(self._nb_sce):
                    weekly_inflow = self._reservoir.hourly_inflow[
                        (week_ind + 1) * constants.RESULTS_INTERVAL_HOURS:
                        (week_ind + 2) * constants.RESULTS_INTERVAL_HOURS, i
                                    ].sum(axis=0)
                    controls = self._cost_function.get_controls(week_ind + 1, i)
                    next_stock = current_stock + weekly_inflow - controls

                    best_value = self.iterate_over_controls_vec(
                        controls=controls,
                        next_stock=next_stock,
                        week_ind=week_ind + 1,
                        sce_ind=i
                    )

                    final_best_value, _, _ = self.iterate_over_stock_levels_vec(
                        best_value=best_value,
                        current_stock_with_inflow=current_stock + weekly_inflow,
                        week_ind=week_ind + 1,
                        sce_ind=i,
                        max_control=controls[-1])

                    bv_sce[i] = final_best_value

                self._bellman_values[week_ind, c // self._reservoir.step] = np.mean(bv_sce)
