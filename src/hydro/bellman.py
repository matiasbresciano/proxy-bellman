import typing

import numpy as np
from scipy.interpolate import interp1d

from hydro.hydro_cost_function import HydroCostFunction
from hydro.hydro_reservoir import HydroReservoir
from bellman import Bellman
import constants


class HydroBellman(Bellman):
    def __init__(self, nb_sce: int, cost_function: HydroCostFunction, reservoir: HydroReservoir):
        super().__init__(nb_sce, cost_function, reservoir)

    def _compute_penalty(self) -> None:
        """
        Returns a piecewise penalty function penalizing deviations outside the weekly lower and upper rule curves.
        Penalties grow linearly beyond ±1% of reservoir capacity from the rule curves.
        """
        # TODO LRI: - voir l'update de Mathias
        self._penalty = np.zeros(shape=constants.RESULTS_SIZE, dtype=typing.Any)

        assert isinstance(self._cost_function, HydroCostFunction)
        assert isinstance(self._reservoir.upper_guide, np.ndarray)
        assert isinstance(self._penalty, np.ndarray)

        for week_idx in range(constants.RESULTS_SIZE - 1):
            max_cost = self._cost_function.max_cost(week_idx)
            penalty = interp1d(
                [
                    self._reservoir.lower_guide[week_idx] - 0.01 * self._reservoir.capacity,
                    self._reservoir.lower_guide[week_idx],
                    self._reservoir.upper_guide[week_idx],
                    self._reservoir.upper_guide[week_idx] + 0.01 * self._reservoir.capacity,
                ],
                [
                    max_cost,
                    0,
                    0,
                    max_cost,
                ],
                fill_value='extrapolate',
            )
            self._penalty[week_idx] = penalty
        week_idx = constants.RESULTS_SIZE - 1
        max_cost = self._cost_function.max_cost(week_idx)
        self._penalty[week_idx] = lambda x: abs(x - self._reservoir.final_level) / (
                10 * self._reservoir.final_level) * 100 * max_cost

    def bellman_function(self, week: int) -> interp1d:
        """
        Returns an interpolated Bellman value function for given week over reservoir stock levels.
        """
        if self._bellman_values is None:
            self._compute_bellman_values()
        assert isinstance(self._bellman_values, np.ndarray)
        return interp1d(
            np.linspace(0, self._reservoir.capacity, 100 // self._reservoir.step + 1),
            self._bellman_values[week],
            kind="linear",
            fill_value="extrapolate"
        )

    def iterate_over_controls_vec(self, controls: np.ndarray, current_stock: float, week_ind: int,
                                  sce_ind: int) -> tuple[float, float, float]:
        assert isinstance(self._penalty, np.ndarray)

        weekly_inflow = self._reservoir.hourly_inflow[
                        week_ind * constants.RESULTS_INTERVAL_HOURS:
                        (week_ind + 1) * constants.RESULTS_INTERVAL_HOURS, sce_ind].sum(axis=0)
        next_stock = current_stock - controls + weekly_inflow
        cost = [self._cost_function.get_cost(week_ind, sce_ind, ctrl) for ctrl in controls]
        penalty = self._penalty[week_ind](next_stock)
        total_value = cost + penalty
        j = int(np.argmin(total_value))
        return float(total_value[j]), float(next_stock[j]), float(controls[j])

    def iterate_over_stock_levels_vec(
            self,
            best_value: tuple[float, float | None, float | None],
            current_stock: float,
            week_ind: int,
            sce_ind: int,
            future_bellman_function: interp1d,
            max_control: float
    ) -> tuple[float, float | None, float | None]:
        """
        Enumerates next stock levels on the 0..100% grid,
        computes implied control, filters infeasible controls, evaluates total value,
        and keeps the best.
        """
        assert isinstance(self._penalty, np.ndarray)

        weekly_inflow = self._reservoir.hourly_inflow[
                        week_ind * constants.RESULTS_INTERVAL_HOURS:
                        (week_ind + 1) * constants.RESULTS_INTERVAL_HOURS, sce_ind].sum(axis=0)
        capacity = float(self._reservoir.capacity)

        next_stock_grid = (np.arange(0, 101, self._reservoir.step, dtype=float) / 100.0) * capacity

        controls = current_stock - next_stock_grid + weekly_inflow
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

        penalty = self._penalty[week_ind](ns)
        cost = [self._cost_function.get_cost(week_ind, sce_ind, c) for c in ctrl]

        total_value = cost + future_bellman_function(ns) + penalty

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
        if self._penalty is None:
            self._compute_penalty()
        assert isinstance(self._penalty, np.ndarray)
        final_penalty = self._penalty[constants.RESULTS_SIZE - 1]
        self._bellman_values[constants.RESULTS_SIZE - 1] = np.array([
            final_penalty((c / 100) * self._reservoir.capacity) for c in range(0, 101, self._reservoir.step)
        ])

        for week_ind in reversed(range(constants.RESULTS_SIZE - 1)):
            future_bellman_function = self.bellman_function(week_ind + 1)

            for c in range(0, 101, self._reservoir.step):
                current_stock = (c / 100) * self._reservoir.capacity
                bv_sce = np.zeros(self._nb_sce)
                for i in range(self._nb_sce):
                    controls = self._cost_function.get_controls(week_ind)

                    best_value = self.iterate_over_controls_vec(
                        controls=controls,
                        current_stock=current_stock,
                        week_ind=week_ind,
                        sce_ind=i
                    )

                    final_best_value, _, _ = self.iterate_over_stock_levels_vec(
                        best_value=best_value,
                        current_stock=current_stock,
                        week_ind=week_ind,
                        sce_ind=i,
                        future_bellman_function=future_bellman_function,
                        max_control=controls[-1])

                    bv_sce[i] = final_best_value

                self._bellman_values[week_ind, c // self._reservoir.capacity] = np.mean(bv_sce)
