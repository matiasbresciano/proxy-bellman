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
        nb_controls = 7 - len(self._reservoir.excluded_week_days) + 1
        self._cost_function = np.zeros(
            shape=(constants.RESULTS_SIZE+1, nb_controls, self._residual_load.shape[1])
        )
        assert isinstance(self._cost_function, np.ndarray)

        # compute first day of the period
        first_day = self._reservoir.get_previous_monday(self._reservoir.first_day)
        if first_day > constants.NB_DAYS - 7 or first_day < self._reservoir.first_day:
            # if first week is not completely in the period, we ignore it
            first_day_second_week = (first_day + 7) % (constants.NB_DAYS+1)
            first_day = first_day_second_week
        day = first_day

        # compute last day of the period
        last_day = self._reservoir.get_previous_monday(self._reservoir.last_day)  # monday of the week of last day

        while day < last_day:
            week_ind = day // 7
            week = self._residual_load[day:day+7]
            week = np.delete(week, self._reservoir.excluded_week_days, axis=0)
            week = np.sort(week, axis=0)
            for control in range(nb_controls):
                # negative cost : gain
                self._cost_function[week_ind, control] = - week[nb_controls-1-control:].sum(axis=0)
            day += 7

        if day + 6 >= self._reservoir.last_day:  # if sunday of last week is out of the period
            week = self._residual_load[day:self._reservoir.last_day+1]
            shape = week.shape
            shape = (7 - shape[0], shape[1])
            if shape[0] != 0:
                week = np.concatenate((week, np.zeros(shape, dtype=np.float64)))
            week = np.delete(week, self._reservoir.excluded_week_days, axis=0)
            week = np.sort(week, axis=0)
            week_ind = day // 7
            for control in range(nb_controls):
                self._cost_function[week_ind, control] = - week[nb_controls - 1 - control:].sum(axis=0)


    def get_cost(self, week_ind: int, sce_ind: int, control: int | float) -> float:
        if self._cost_function is None:
            self._compute_cost_function()
        assert isinstance(self._cost_function, np.ndarray)
        return float(self._cost_function[week_ind, int(control), sce_ind])

