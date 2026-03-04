from cost_function import CostFunction
from hydro.hydro_reservoir import HydroReservoir
import constants

import numpy as np
from scipy.interpolate import interp1d


class HydroCostFunction(CostFunction):
    """Cost function class for hydro. Inherits from CostFunction

    Computes and provides cost values and maximum_cost over a week (for penalty values)

    Attributes:
        turb_threshold (int): number of values on which the cost function is computed (default is 25)
        alpha (int): parameter for the computation of the costs value and the turbine vs pumping ratio
        __max_cost (np.array): maximum cost for each week to use as penalty
    """
    turb_threshold: int
    alpha: int
    __max_cost: np.ndarray[tuple[int, int], np.dtype[np.number]] | None

    def __init__(self, residual_load: np.ndarray, reservoir: HydroReservoir, turb_threshold: int = 25,
                 alpha: int = 2):
        super().__init__(residual_load, reservoir)
        self.turb_threshold = turb_threshold
        self.alpha = alpha
        self.__max_cost = None

    def _compute_cost_function(self) -> None:
        """
        Compute and store cost-related interpolators for all weeks and scenarios.
        """
        self._cost_function = np.empty(
            (constants.RESULTS_SIZE, self._residual_load.shape[1]),
            dtype=object
        )
        self.__max_cost = np.zeros(
            (constants.RESULTS_SIZE, self._residual_load.shape[1], self.turb_threshold),
            dtype=np.float64
        )
        self._controls = np.zeros(shape=constants.RESULTS_SIZE, dtype=object)
        for w in range(self._cost_function.shape[0]):
            for s in range(self._cost_function.shape[1]):
                self.__stage_cost_function(w, s)

    def __stage_cost_function(self, week: int, scenario: int) -> None:
        """
        Compute the cost for a given week and scenario,
        as functions of the weekly energy control.

        Returns:
            Interpolator (scipy interp1d): cost(control),
        """
        assert isinstance(self._reservoir, HydroReservoir)  # to avoid typing errors
        assert isinstance(self._controls, np.ndarray)  # to avoid typing errors
        assert isinstance(self.__max_cost, np.ndarray)  # to avoid typing errors
        assert isinstance(self._cost_function, np.ndarray)  # to avoid typing errors

        weekly_net_load = self._residual_load[week * 168:(week + 1) * 168, scenario]
        max_hourly_turb = self._reservoir.hourly_max_turb[week * 168:(week + 1) * 168]
        max_hourly_pump = self._reservoir.hourly_max_pump[week * 168:(week + 1) * 168]
        null_pump = np.allclose(self._reservoir.hourly_max_pump, 0)

        low = np.min(weekly_net_load - max_hourly_turb)
        raw_high = (np.max(weekly_net_load + max_hourly_pump) *
                    (self._reservoir.turb_efficiency / self._reservoir.pump_efficiency) ** (
                    1 / (self.alpha - 1)))
        high = np.max(weekly_net_load) if null_pump else raw_high

        turb_thresholds = np.unique(np.linspace(low, high, self.turb_threshold))

        weekly_control, costs = self.__compute_control_with_thresholds(
            turb_thresholds=turb_thresholds,
            weekly_net_load=weekly_net_load,
            max_hourly_turb=max_hourly_turb,
            max_hourly_pump=max_hourly_pump,
            null_pump=null_pump
        )

        self._controls[week] = weekly_control.copy()

        idx = np.argsort(weekly_control)
        weekly_control = weekly_control[idx]
        costs = costs[idx]

        self.__max_cost[week, scenario] = float(max(costs[0], costs[-1]))

        self._cost_function[week, scenario] = interp1d(weekly_control, costs, fill_value="extrapolate")

    def __compute_control_with_thresholds(
            self,
            turb_thresholds: np.ndarray,
            weekly_net_load: np.ndarray,
            max_hourly_turb: np.ndarray,
            max_hourly_pump: np.ndarray,
            null_pump: bool
    ) -> tuple:
        """
        Compute weekly control and cost using
        threshold-based hourly policies.

        Each policy is defined by a turbining threshold applied to the hourly
        net load: turbining is activated when the net load is above this
        threshold and limited by the maximum hourly turbining capacity.
        When pumping is allowed, a corresponding pumping threshold is defined as:
            pump_threshold =
                turb_threshold                                      if turb_threshold < 0
                turb_threshold * (eta_pump / eta_turb)^(1/(alpha-1)) otherwise

        Pumping is activated when the net load is below the pumping threshold,
        within the limits of the maximum hourly pumping capacity. Hourly
        turbining and pumping are aggregated over the week to compute the
        net weekly control and the associated transition cost.
        """
        assert isinstance(self._reservoir, HydroReservoir)  # to avoid typing errors

        nl = weekly_net_load[None, :]
        mt = max_hourly_turb[None, :]
        mp = max_hourly_pump[None, :]
        tt = turb_thresholds[:, None]

        # Turbining clip
        clipped = np.minimum(nl, np.maximum(tt, nl - mt))
        turb = nl - clipped

        # Pumping
        if null_pump:
            pump = np.zeros_like(clipped)
        else:
            ratio = (self._reservoir.pump_efficiency / self._reservoir.turb_efficiency) ** (1 / (self.alpha - 1))

            pump_thresholds = np.where(
                turb_thresholds < 0,
                turb_thresholds,
                ratio * turb_thresholds
            )

            pt = pump_thresholds[:, None]
            potential_pump = pt - clipped
            mask = clipped < pt

            pump = np.where(mask, np.minimum(potential_pump, mp), 0.0)

        net_after = clipped + pump

        # Aggregations
        weekly_control = (turb * self._reservoir.turb_efficiency - pump * self._reservoir.pump_efficiency).sum(axis=1)
        costs = (np.abs(net_after) ** self.alpha).sum(axis=1)

        return weekly_control, costs

    def get_cost(self, week_ind: int, sce_ind: int, control: float | int) -> float:
        """Returns the cost value associated to a week, scenario and control.

        Returns the cost value associated to a week, scenario and control. If needed, launches the
        cost function computation first.

        Returns:
            cost value corresponding to the parameters
        """
        if self._cost_function is None:
            self._compute_cost_function()

        assert isinstance(self._cost_function, np.ndarray)
        cost_function: interp1d = self._cost_function[week_ind, sce_ind]
        assert isinstance(cost_function, interp1d)
        return cost_function(control)

    def max_cost(self, week: int) -> float:
        """
        Compute a conservative maximum cost for a given week.

        For each scenario at the given week, this inspects the interpolated
        cost function c_w^s(u) at the two extreme control values available
        in its grid (controls[0] and controls[-1]), and returns the minimum over
        both extremes and all scenarios:

            min_cost = min_s  min( c_w^s(u_min), c_w^s(u_max) )

        Args:
            week (int): Week index.

        Returns:
            float: minimum cost of the stage cost for the given week across scenarios.
        """
        if self.__max_cost is None:
            self._compute_cost_function()
        assert isinstance(self.__max_cost, np.ndarray)
        return self.__max_cost[week].max()
