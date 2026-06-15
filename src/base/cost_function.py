"""
Base modelisation of the cost function used to compute a trajectory
"""

import typing
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d

from base.reservoir import Reservoir


class CostFunction(ABC):
    """This abstract class is a model for cost functions

    Attributes:
        _residual_load (np.ndarray): residual_load of the different scenarios provided (hourly)
        _reservoir (Reservoir): Reservoir describing the stock
        _cost_function (np.ndarray): cost function computed for the different controls and stock levels possible.
            Type is np.ndarray[tuple[int, int], np.dtype[interp1d]] for continuous use cases,
            np.ndarray[tuple[int, int, int], np.dtype[np.float64]] for discrete use cases.
        _controls (np.ndarray): controls value for which costs are computed
    """

    def __init__(self, residual_load: np.ndarray, reservoir: Reservoir) -> None:
        self._residual_load: np.ndarray = residual_load
        self._reservoir: Reservoir = reservoir
        self._cost_function: (np.ndarray[tuple[int, int], np.dtype[interp1d]] |
                              np.ndarray[tuple[int, int, int], np.dtype[np.float64]] |
                              None
                              ) = None
        self._controls: np.ndarray[tuple[int, int, int], np.dtype[np.float64]] | None = None

    @abstractmethod
    def _compute_cost_function(self) -> None:
        """private method to compute the cost function"""
        pass

    @abstractmethod
    def get_cost(self, week_ind: int, sce_ind: int, control: float | int) -> float:
        """get the cost value linked to a week, a control and a scenario"""
        pass
