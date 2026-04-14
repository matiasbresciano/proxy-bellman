from abc import ABC, abstractmethod
import numpy as np
import typing
import math

from cost_function import CostFunction
from reservoir import Reservoir
import constants

"""
Base for the Bellman values computation classes
"""


class Bellman(ABC):
    """This abstract class is a model for Bellman values computation classes

    Attributes:
        _nb_sce (int): number of scenarii
        _reservoir (Reservoir): Reservoir describing the stock
        _cost_function (CostFunction): gain function to use for computing bellman values
        _bellman_values (np.ndarray): the value associated to each possible stock level for each week
        _usage_value (np.ndarray): the usage value associated to each week and each possible stock level
    """
    _nb_sce: int
    _cost_function: CostFunction
    _reservoir: Reservoir
    _bellman_values: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None
    _usage_value: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None

    def __init__(self, nb_sce: int, cost_function: CostFunction, reservoir: Reservoir) -> None:
        self._nb_sce = nb_sce
        self._cost_function = cost_function
        self._reservoir = reservoir
        self._bellman_values = None
        self._usage_value = None

    @abstractmethod
    def _compute_bellman_values(self) -> None:
        pass

    def _compute_usage_values(self) -> None:
        self._usage_value = np.zeros(
            shape=(constants.RESULTS_SIZE, math.floor(self._reservoir.capacity/self._reservoir.step)),
            dtype=np.float64)
        assert isinstance(self._bellman_values, np.ndarray)  # to avoid typing errors
        for w in range(constants.RESULTS_SIZE):
            for c in range(math.ceil(self._reservoir.capacity)):
                self._usage_value[w, c - 1] = self._bellman_values[w, c] - self._bellman_values[w, c - 1]

    def get_bellman_values(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        if self._bellman_values is None:
            self._compute_bellman_values()
        assert isinstance(self._bellman_values, np.ndarray)
        return self._bellman_values

    @abstractmethod
    def get_penalty(self, week: int, stock: float|int) -> float:
        """Returns the computed penalty for given week and stock"""
        pass

    def get_usage_values(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        if self._usage_value is None:
            self._compute_usage_values()
        assert isinstance(self._usage_value, np.ndarray)
        return self._usage_value
