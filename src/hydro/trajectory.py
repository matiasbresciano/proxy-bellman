import numpy as np

from hydro.bellman import HydroBellman
from hydro.reservoir import HydroReservoir
from hydro.cost_function import HydroCostFunction
from trajectory import Trajectory
import constants


class HydroTrajectory(Trajectory):
    """Trajectory class for hydro. Inherits from Trajectory

    Computes and provides trajectories and control values for each scenario and each week

    Attributes:
        inflow_adjust_overflow (float): factor to modulate how important it is to respect guidelines
    """
    inflow_adjust_overflow: None | np.ndarray

    def __init__(self, nb_sce: int, reservoir: HydroReservoir, cost_function: HydroCostFunction,
                 bellman: HydroBellman):
        """
        Initialize OptimalTrajectories with a BellmanValuesProxy instance.
        Prepares data and computes optimal trajectories.
        """
        super().__init__(nb_sce, reservoir, cost_function, bellman)
        self.inflow_adjust_overflow = None

    def _compute_trajectories(self) -> None:
        """
        Compute optimal reservoir trajectories and controls schedules
        for all scenarios and weeks using Bellman values, penalties and inflows.
        Adjusts hourly inflows to avoid overflow or negative stock with external method.
        """
        self._trajectories = np.zeros((self._nb_sce, constants.RESULTS_SIZE))
        self._controls = np.zeros_like(self._trajectories)
        self.optimal_turb = np.zeros_like(self._trajectories)
        self.optimal_pump = np.zeros_like(self._trajectories)
        self.inflow_adjust_overflow = np.zeros((constants.RESULTS_SIZE, self._nb_sce, constants.RESULTS_INTERVAL_HOURS))

        assert isinstance(self._reservoir, HydroReservoir)
        assert isinstance(self._bellman, HydroBellman)

        for s in range(self._nb_sce):
            current_stock = self._reservoir.initial_level
            
            for w in range(constants.RESULTS_SIZE):
                hourly_inflow = self._reservoir.hourly_inflow[w * 168:(w + 1) * 168, s]
                weekly_inflow: float = hourly_inflow.sum()

                controls = self._cost_function.get_controls(w, s)

                max_control = self.adjust_hourly_inflow_overflow(scenario=s, 
                                                                 week=w, 
                                                                 stock_init=current_stock, 
                                                                 inflow=hourly_inflow)                
                if max_control < controls[-1]:
                    controls = controls[controls < max_control]
                    controls = np.concatenate([controls, [max_control]])

                weekly_inflow -= np.sum(self.inflow_adjust_overflow[w, s])

                next_stock = current_stock + weekly_inflow - controls
                feasible = (next_stock >= 0) & (next_stock <= self._reservoir.capacity+1e-6)
                controls = controls[feasible]

                best_value = self._bellman.iterate_over_controls_vec(
                    controls=controls,
                    next_stock=next_stock[feasible],
                    week_ind=w,
                    sce_ind=s,
                    exact_ctrls=False
                )

                __, final_best_stock, final_best_control = self._bellman.iterate_over_stock_levels_vec(
                    best_value=best_value,
                    current_stock_with_inflow=current_stock + weekly_inflow,
                    week_ind=w,
                    sce_ind=s,
                    max_control=controls[-1])

                self._trajectories[s, w] = final_best_stock
                self._controls[s, w] = final_best_control
                assert isinstance(final_best_stock, float)
                current_stock = final_best_stock

    def adjust_hourly_inflow_overflow(
        self,
        scenario: int,
        week: int,
        stock_init: float,
        inflow: np.ndarray
    ) -> float:
        """
        Adjust hourly inflow to mitigate overflow or negative stock by detecting violations and recording adjustments.
        Returns the maximum feasible control (turbining) for the week after adjustments and taking in account potential overflow or negative stock during the week.
        """
        assert isinstance(self._reservoir, HydroReservoir)
        assert isinstance(self.inflow_adjust_overflow, np.ndarray)
        cap = float(self._reservoir.capacity)

        turb = self._reservoir.hourly_max_turb[week * 168:(week + 1) * 168]
        max_control = turb * self._reservoir.turb_efficiency

        net_hourly_turb = inflow - max_control

        stock = float(stock_init)

        for h in range(168):
            stock += float(net_hourly_turb[h])

            if stock > cap:
                overflow = stock - cap

                self.inflow_adjust_overflow[week, scenario, h] = overflow

                net_hourly_turb[h] -= overflow
                stock = cap

            if stock < 0.0:
                neg = stock  

                max_control[h] += neg

                net_hourly_turb[h] -= neg 
                stock = 0.0

        return float(np.sum(max_control))
