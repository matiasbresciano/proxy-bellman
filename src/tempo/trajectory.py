import numpy as np

from tempo.bellman import TempoBellman
from tempo.reservoir import TempoReservoir
from tempo.cost_function import TempoCostFunction
from trajectory import Trajectory
import constants


class TempoTrajectory(Trajectory):
    """Trajectory class for hydro. Inherits from Trajectory

    Computes and provides trajectories and control values for each scenario and each week
    """
    daily_trajectory: None | np.ndarray

    def __init__(self, nb_sce: int, reservoir: TempoReservoir, cost_function: TempoCostFunction, bellman: TempoBellman):
        super().__init__(nb_sce, reservoir, cost_function, bellman)
        self.daily_trajectory = None

    def _compute_trajectories(self) -> None:
        """
        Compute stock and control trajectories over all scenarios and weeks,
        considering possible constraints from reduced stock trajectories.

        Vectorized over scenarios and controls (loop only over weeks).
        """
        assert isinstance(self._reservoir, TempoReservoir)
        self._trajectories = np.zeros((self._nb_sce, constants.RESULTS_SIZE), dtype=int)
        self._trajectories[:, 0] = self._reservoir.capacity
        self.daily_trajectory = np.zeros((self._nb_sce, constants.NB_DAYS + 1), dtype=int)
        self.daily_trajectory[:, 0] = self._reservoir.capacity
        self._controls = np.zeros_like(self._trajectories)
        nb_controls = 7 - len(self._reservoir.excluded_week_days)
        controls = np.arange(nb_controls, dtype=int)
        active = np.ones(self._nb_sce, dtype=bool)
        for week_ind in range(constants.RESULTS_SIZE):
            # If week>0 and previous stock == 0 => set rest to 0 and stop updating this scenario
            if week_ind >= 1:
                prev_stock = self._trajectories[:, week_ind - 1]
                just_deactivated = active & (prev_stock == 0)
                if np.any(just_deactivated):
                    self._trajectories[just_deactivated, week_ind:] = 0.0
                    # controls already 0 by default; keep them.
                    active[just_deactivated] = False

            if not np.any(active):
                break
            # Work only on active scenarios
            list_sce = np.where(active)[0]
            if week_ind > 0:
                active_prev_stock = self._trajectories[list_sce, week_ind - 1].astype(int)
            else:
                active_prev_stock = self._trajectories[:, week_ind]

            # Precompute gains for this week for all active scenarios and all controls: (S_act, A)
            costs = np.asarray([[self._cost_function.get_cost(week_ind, sce, ctrl)
                                 for ctrl in controls]
                                for sce in list_sce])
            if not costs.any():
                self._trajectories[list_sce, week_ind] = active_prev_stock
                continue

            # Indices of future value: stock after applying control (broadcast) => (S_act, A)
            next_stocks = active_prev_stock[:, None] - controls[None, :]
            # Future BV and penalty evaluated for all (scenario, control)
            future_value = self._bellman.get_bellman_values()[week_ind, next_stocks]
            future_value = np.where(next_stocks < 0, 0, future_value)
            penalty = np.asarray([[self._bellman.get_penalty(week_ind, stc)
                                   for stc in sce]
                                  for sce in next_stocks])
            total_values = - costs + future_value + penalty
            best_controls = controls[np.argmax(total_values, axis=1)].astype(float)  # (S_act,)

            self._trajectories[list_sce, week_ind] = active_prev_stock - best_controls
            self._controls[list_sce, week_ind] = best_controls
