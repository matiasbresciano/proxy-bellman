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
        _cost_function (CostFunction): gain function to use for computing bellman values
        _bellman (np.ndarray): bellman values
        _trajectories (np.ndarray): for each scenario, for each week the computed stock level
        _controls (np.ndarray): for each scenario, for each week, the amount used
    """
    _reservoir: Reservoir
    _cost_function: CostFunction
    _bellman: Bellman
    _trajectories: np.ndarray[tuple[int, int], np.dtype[np.number]]|None
    _controls: np.ndarray[tuple[int, int], np.dtype[np.number]]|None
    nb_sce: int

    def __init__(self, nb_sce: int,  reservoir: Reservoir, gain_function: CostFunction, bellman: Bellman) -> None:
        self._reservoir = reservoir
        self._cost_function = gain_function
        self._bellman = bellman
        self._trajectories = None
        self._controls = None
        self._nb_sce = nb_sce

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



