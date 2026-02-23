from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d

"""
Base modelisation of the gain function used to compute a trajectory
"""


class GainFunction(ABC):
    """This abstract class is a model for gain functions

    Attributes:
        residual_load (np.ndarray): residual_load of the different scenarios provided (hourly)
        gain_function (np.ndarray): gain function computed for the different controls and stock levels possible
    """
    residual_load: np.ndarray
    gain_function: np.ndarray | interp1d

    def __init__(self, residual_load) -> None:
        self.residual_load = residual_load

    @abstractmethod
    def _compute_gain_function(self) -> None:
        """private method to compute the gain function"""
        pass

    @abstractmethod
    def get_gain(self, stock: float | int, control: float | int, sce_id: int) -> float:
        """get the gain value linked to a stock level, a control and a scenario"""
        pass
