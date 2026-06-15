"""
Base for the trajectories computation classes
"""

from abc import ABC, abstractmethod
import numpy as np

from base.reservoir import Reservoir
from base.cost_function import CostFunction
from base.bellman import Bellman
import constants


class Trajectory(ABC):
    """This abstract class is a model for trajectories values computation classes

    Attributes:
        _reservoir (Reservoir): Reservoir describing the stock
        _cost_function (CostFunction): gain function to use for computing bellman values
        _bellman (np.ndarray): bellman values
        _trajectories (np.ndarray): for each scenario, for each week the computed stock level
        _controls (np.ndarray): for each scenario, for each week, the amount used
        _list_sce (np.ndarray): list of scenarii to consider
    """
    def __init__(self, list_sce: np.ndarray, reservoir: Reservoir, cost_function: CostFunction, bellman: Bellman) -> None:
        self._reservoir: Reservoir = reservoir
        self._cost_function: CostFunction = cost_function
        self._bellman: Bellman = bellman
        self._trajectories: np.ndarray[tuple[int, int], np.dtype[np.number]] | None = None
        self._controls: np.ndarray[tuple[int, int], np.dtype[np.number]] | None = None
        self._list_sce: np.ndarray = list_sce

    @abstractmethod
    def _compute_trajectories(self) -> None:
        pass

    def get_trajectories(self) -> np.ndarray[tuple[int, int], np.dtype[np.number]]:
        if self._trajectories is None:
            self._compute_trajectories()
        assert isinstance(self._trajectories, np.ndarray)
        return self._trajectories

    def get_controls(self) -> np.ndarray[tuple[int, int], np.dtype[np.number]]:
        if self._controls is None:
            self._compute_trajectories()
        assert isinstance(self._controls, np.ndarray)
        return self._controls



