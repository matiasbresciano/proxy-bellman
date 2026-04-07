import numpy as np
from scipy.interpolate import interp1d

from tempo.cost_function import TempoCostFunction
from tempo.reservoir import TempoReservoir
from bellman import Bellman
import constants


class TempoBellman(Bellman):
    """Bellman class for tempo. Inherits from Bellman

    Computes and provides bellman values and penalties for each week

    Attributes:
        c_var: percentage of scenario to consider at each week. if <1, most favorable scenario are ignored
    """
    c_var: float

    def __init__(self, nb_sce: int, cost_function: TempoCostFunction, reservoir: TempoReservoir, c_var: float = 1.0):
        super().__init__(nb_sce, cost_function, reservoir)
        self.c_var = c_var

    def _compute_bellman_values(self) -> None:
        self._bellman_values = np.zeros(shape=(constants.RESULTS_SIZE+1, int(self._reservoir.capacity) + 1), dtype=np.float64)
        assert isinstance(self._reservoir, TempoReservoir)
        nb_controls = 7 + 1 - len(self._reservoir.excluded_week_days)
        controls = np.arange(nb_controls, dtype=int)

        for week_ind in reversed(range(constants.RESULTS_SIZE)):
            costs = np.asarray([[self._cost_function.get_cost(week_ind+1, sce, int(ctrl))
                                 for ctrl in controls]
                                for sce in range(self._nb_sce)])
            if not costs.any():
                continue

            cutoff_index = int((1 - self.c_var) * self._nb_sce)

            for stock in range(int(self._reservoir.capacity) + 1):
                next_stock = stock - controls
                next_stock = np.where(next_stock < 0, 0, next_stock)
                future_value = self._bellman_values[week_ind + 1, next_stock]  # shape (A,)
                penalty = np.asarray([self.get_penalty(week_ind, int(stc)) for stc in next_stock])

                nb_feasible = len(next_stock)
                # Total value for each (scenario, control): shape (S, A)
                total_values = -costs[:, :nb_feasible] + future_value[None, :] + penalty[None, :]

                # Best value per scenario: shape (S,)
                best_per_scenario = np.max(total_values, axis=1)

                # CVaR tail-mean over scenarios
                sorted_bv = np.sort(best_per_scenario)
                self._bellman_values[week_ind, stock] = float(np.mean(sorted_bv[cutoff_index:]))

    def get_penalty(self, week: int, stock: int|float) -> float:
        assert isinstance(self._reservoir.upper_guide, np.ndarray)
        penalty = interp1d([
                self._reservoir.lower_guide[week] - 1,
                self._reservoir.lower_guide[week],
                self._reservoir.upper_guide[week],
                self._reservoir.upper_guide[week] + 1,
            ],
            [-1e9, 0, 0, -1e9],
            kind='linear', fill_value='extrapolate')
        # Alternative no penalty: penalty = lambda x: 0
        return penalty(stock)

