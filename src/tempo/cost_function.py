import numpy as np

from cost_function import CostFunction
from tempo.reservoir import TempoReservoir
import constants


class TempoCostFunction(CostFunction):
    """Cost function class for tempo. Inherits from CostFunction

    Computes and provides cost values and maximum_cost over a week (for penalty values)
    """
    def __init__(self, residual_load: np.ndarray, reservoir: TempoReservoir):
        super().__init__(residual_load, reservoir)

    def _compute_cost_function(self) -> None:
        """
        Compute gain for each week index, control level, and scenario.
        Gains are sum of the top 'control' daily net loads in the considered week,
        limited by max_control.
        """
        assert isinstance(self._reservoir, TempoReservoir)
        nb_controls = 7 - len(self._reservoir.excluded_week_days)
        self._cost_function = np.zeros(
            shape=(constants.RESULTS_SIZE+1, nb_controls, self._residual_load.shape[1])
        )

        first_day = self._reservoir.get_previous_monday(self._reservoir.first_day)
        if first_day > constants.NB_DAYS - 7:  # if first day is in last week of the year
            first_day_second_week = (first_day + 7) % (constants.NB_DAYS+1)
            week = np.concatenate((self._residual_load[first_day:],
                                  self._residual_load[:first_day_second_week]),
                                  axis=0)
            week = np.delete(week, self._reservoir.excluded_week_days, axis=0)
            week = np.sort(week, axis=0)
            for control in range(nb_controls):
                self._cost_function[0, control] = - week[nb_controls-control:].sum(axis=0)
            first_day = first_day_second_week
        last_day = self._reservoir.get_previous_monday(self._reservoir.last_day) + 6  # sunday of the week of last day
        if last_day > constants.NB_DAYS:  # if last day is in first week of next year
            # last week loops over first week and has already been computed
            last_day = self._reservoir.get_previous_monday(self._reservoir.last_day) - 1
        for day in range(first_day, last_day, 7):
            week_ind = (day + 6) % (constants.NB_DAYS+1) // 7
            week = self._residual_load[day:day+7]
            week = np.delete(week, self._reservoir.excluded_week_days, axis=0)
            week = np.sort(week, axis=0)
            for control in range(nb_controls):
                # negative cost : gain
                self._cost_function[week_ind, control] = - week[nb_controls-control:].sum(axis=0)

    def get_cost(self, week_ind: int, sce_ind: int, control: int) -> float:
        if self._cost_function is None:
            self._compute_cost_function()
        assert isinstance(self._cost_function, np.ndarray)
        return float(self._cost_function[week_ind, control, sce_ind])

