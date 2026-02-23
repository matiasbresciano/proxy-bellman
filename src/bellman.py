from abc import ABC, abstractmethod
import numpy as np
import typing

from gain_function import GainFunction
from reservoir import Reservoir
import constants

"""
Base for the Bellman values computation classes
"""


class Bellman(ABC):
    """This abstract class is a model for Bellman values computation classes

    Attributes:
        _reservoir (Reservoir): Reservoir describing the stock
        _gain_function (GainFunction): gain function to use for computing bellman values
        _bellman_values (np.ndarray): the value associated to each possible stock level for each week
        _penalty (np.ndarray): for each week, the interp1d function to compute the penalty associated to a stock level
        _usage_value (np.ndarray): the usage value associated to each week and each possible stock level
    """
    _gain_function: GainFunction
    _reservoir: Reservoir
    _bellman_values: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None
    _penalty: np.ndarray[tuple[int], np.dtype[typing.Any]] | None
    _usage_value: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None

    def __init__(self, gain_function: GainFunction, reservoir: Reservoir) -> None:
        self._gain_function = gain_function
        self._reservoir = reservoir
        self._penalty = None

    @abstractmethod
    def _compute_bellman_values(self) -> None:
        pass

    @abstractmethod
    def _compute_penalty(self) -> None:
        pass

    def _compute_usage_values(self) -> None:
        self._usage_value = np.zeros((constants.RESULTS_SIZE, self._reservoir.capacity))
        for w in range(constants.RESULTS_SIZE):
            for c in range(self._reservoir.capacity):
                self._usage_value[w, c - 1] = self._bellman_values[w, c] - self._bellman_values[w, c - 1]

    def get_bellman_values(self) -> np.ndarray:
        if self._bellman_values is None:
            self._compute_bellman_values()
        return self._bellman_values

    def get_penalties(self) -> np.ndarray:
        """Returns the array containing all penalty functions np.array(interp1D)"""
        if self._penalty is None:
            self._compute_penalty()
        return self._penalty

    def get_usage_values(self) -> np.ndarray:
        if self._usage_value is None:
            self._compute_usage_values()
        return self._usage_value
