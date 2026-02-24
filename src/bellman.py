from abc import ABC, abstractmethod
import numpy as np
import typing
import math

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
    _bellman_values: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    _penalty: np.ndarray[tuple[int], np.dtype[typing.Any]]
    _usage_value: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    def __init__(self, gain_function: GainFunction, reservoir: Reservoir) -> None:
        self._gain_function = gain_function
        self._reservoir = reservoir
        self._penalty = np.array([None])
        self._bellman_values = np.zeros(shape=(1, 1), dtype=np.float64)
        self._usage_value = np.zeros(shape=(1, 1), dtype=np.float64)

    @abstractmethod
    def _compute_bellman_values(self) -> None:
        pass

    @abstractmethod
    def _compute_penalty(self) -> None:
        pass

    def _compute_usage_values(self) -> None:
        self._usage_value = np.zeros(
            shape=(constants.RESULTS_SIZE, math.floor(self._reservoir.capacity/self._reservoir.step)),
            dtype=np.float64)
        for w in range(constants.RESULTS_SIZE):
            for c in range(math.ceil(self._reservoir.capacity)):
                self._usage_value[w, c - 1] = self._bellman_values[w, c] - self._bellman_values[w, c - 1]

    def get_bellman_values(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        if self._bellman_values.shape[0] != constants.RESULTS_SIZE:
            self._compute_bellman_values()
        return self._bellman_values

    def get_penalties(self) -> np.ndarray[tuple[int], np.dtype[typing.Any]]:
        """Returns the array containing all penalty functions np.array(interp1D)"""
        if self._penalty.shape[0] != constants.RESULTS_SIZE:
            self._compute_penalty()
        return self._penalty

    def get_usage_values(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        if self._usage_value.shape[0] != constants.RESULTS_SIZE:
            self._compute_usage_values()
        return self._usage_value
