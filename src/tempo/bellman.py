import numpy as np
from scipy.interpolate import interp1d

from tempo.cost_function import TempoCostFunction
from tempo.reservoir import TempoReservoir
from base.bellman import Bellman
import constants


class TempoBellman(Bellman):
    """Bellman class for tempo. Inherits from Bellman.

    Computes and provides bellman values and penalties for each week.

    Attributes:
        c_var: percentage of scenario to consider at each week. if <1, most favorable scenario are ignored.
    """

    def __init__(self, list_sce: np.ndarray, cost_function: TempoCostFunction, reservoir: TempoReservoir, c_var: float = 1.0):
        super().__init__(list_sce, cost_function, reservoir)
        self.c_var = c_var

    def get_penalty(self, week: int, stock: int|float) -> float:
        """Returns the penalty associated to a certain stock value for a given week.
        """
        assert isinstance(self._reservoir.upper_guide, np.ndarray)
        penalty = interp1d([
                self._reservoir.lower_guide[week] - 1,
                self._reservoir.lower_guide[week],
                self._reservoir.upper_guide[week],
                self._reservoir.upper_guide[week] + 1,
            ],
            [1e9, 0, 0, 1e9],
            kind='linear', fill_value='extrapolate')
        # Alternative no penalty: penalty = lambda x: 0
        return penalty(stock)

