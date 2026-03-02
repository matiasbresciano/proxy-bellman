from abc import ABC, abstractmethod
import numpy as np

from reservoir import Reservoir
from cost_function import CostFunction
from bellman import Bellman
import constants

"""
Base for the trajectories computation classes
"""


class Trajectory(ABC):
    """This abstract class is a model for trajectories values computation classes

    Attributes:
        _reservoir (Reservoir): Reservoir describing the stock
        _gain_function (CostFunction): gain function to use for computing bellman values
        _bellman (np.ndarray): bellman values
        _trajectories (np.ndarray): for each scenario, for each week the computed stock level
        _controls (np.ndarray): for each scenario, for each week, the amount used
    """
    _reservoir: Reservoir
    _gain_function: CostFunction
    _bellman: Bellman
    _trajectories: np.ndarray[tuple[int, int], np.dtype[np.number]]
    _controls: np.ndarray[tuple[int, int], np.dtype[np.number]]

    def __init__(self, reservoir: Reservoir, gain_function: CostFunction, bellman: Bellman) -> None:
        self._reservoir = reservoir
        self._gain_function = gain_function
        self._bellman = bellman
        self._trajectories = np.zeros(shape=(1, 1), dtype=np.float64)
        self._controls = np.zeros(shape=(1, 1), dtype=np.float64)

    @abstractmethod
    def _compute_trajectories(self) -> None:
        pass

    def get_trajectories(self) -> np.ndarray[tuple[int, int], np.dtype[np.number]]:
        if self._trajectories.shape[0] != constants.RESULTS_SIZE:
            self._compute_trajectories()
        return self._trajectories

    def get_controls(self) -> np.ndarray[tuple[int, int], np.dtype[np.number]]:
        if self._controls.shape[0] != constants.RESULTS_SIZE:
            self._compute_trajectories()
        return self._controls



