import pytest
import numpy as np

from tempo.cost_function import TempoCostFunction
from tempo.reservoir import TempoReservoir
import constants


def test_rand_net_load():
    nb_sce = 2

    def gain_for_week_control_and_scenario(daily_net_load, week_index: int, control: int, scenario: int, max_control:int) -> float:
        # previous implementation
        week_start = week_index * 7
        week_end = week_start + 7

        daily_load_week = daily_net_load[week_start:week_end, scenario]
        # Sort and take top 'max_control' values in descending order
        daily_load_week = np.sort(daily_load_week[:max_control])[::-1]

        gain = np.sum(daily_load_week[:control])
        return gain

    residual_load = np.random.rand(constants.NB_DAYS + 1, nb_sce)*1000
    res = TempoReservoir()
    cost = TempoCostFunction(residual_load, res)
    for i in range(constants.RESULTS_SIZE):
        if res.get_previous_monday(7*i) < res.last_day < res.get_previous_monday(7*i) + 7:
            c = cost.get_cost(i, 0, 3)
            expected = -gain_for_week_control_and_scenario(residual_load, i, 3, 0, 6)
            assert c >= expected, "week: " + str(i) + " cost : " + str(c)

            c = cost.get_cost(i, 1, 1)
            expected = -gain_for_week_control_and_scenario(residual_load, i, 1, 1, 6)
            assert c >= expected, "week: " + str(i) + " cost : " + str(c)
        elif res.get_previous_monday(7*i) < res.first_day or res.get_previous_monday(7*i) > res.last_day:
            c = cost.get_cost(i, 0, 1)
            assert c == 0, "week: " + str(i) + " cost : " + str(c)
            c = cost.get_cost(i, 1, 1)
            assert c == 0, "week: " + str(i) + " cost : " + str(c)
        else:
            c = cost.get_cost(i, 0, 3)
            expected = -gain_for_week_control_and_scenario(residual_load, i, 3, 0, 5)
            assert c == pytest.approx(expected), "week: " + str(i) + " cost : " + str(c)

            c = cost.get_cost(i, 1, 1)
            expected = -gain_for_week_control_and_scenario(residual_load, i, 1, 1, 5)
            # assert c == pytest.approx(expected), "week: " + str(i) + " cost : " + str(c)

            assert cost.get_cost(i, 1, 0) == 0


def test_different_week_day():
    for d in range(7):
        nb_sce = 1
        residual_load = np.random.rand(constants.NB_DAYS + 1, nb_sce)*1000
        res = TempoReservoir()
        res.week_day_first_september = d
        cost = TempoCostFunction(residual_load, res)

        for i in range(constants.RESULTS_SIZE):
            if res.get_previous_monday(7*i+(7-d)%7) < res.first_day or res.get_previous_monday(7*i+(7-d)%7) > res.last_day:
                c = cost.get_cost(i, 0, 1)
                assert c == 0, "week: " + str(i) + " cost : " + str(c) + " week_day 1st september : " + str(d)
            else:
                c = cost.get_cost(i, 0, 3)
                assert c != 0, "week: " + str(i) + " week_day 1st september : " + str(d)


def test_rand_net_load_white():
    nb_sce = 2

    def gain_for_week_control_and_scenario(daily_net_load, week_index: int, control: int, scenario: int, max_control:int) -> float:
        # previous implementation
        week_start = week_index * 7  # If year begins on monday 1st September
        week_end = week_start + 7

        daily_load_week = daily_net_load[week_start:week_end, scenario]
        # Sort and take top 'max_control' values in descending order
        daily_load_week = np.sort(daily_load_week[:max_control])[::-1]

        gain = np.sum(daily_load_week[:control])
        return gain

    residual_load = np.random.rand(constants.NB_DAYS + 1, nb_sce)*1000
    res = TempoReservoir(first_day=0, last_day=constants.NB_DAYS, excluded_week_days=np.asarray([6]))
    cost = TempoCostFunction(residual_load, res)
    for i in range(constants.RESULTS_SIZE):
        if res.get_previous_monday(7*i) < res.last_day < res.get_previous_monday(7*i) + 7:
            c = cost.get_cost(i, 0, 3)
            expected = -gain_for_week_control_and_scenario(residual_load, i, 3, 0, 6)
            assert c <= expected, "week: " + str(i) + " cost : " + str(c)

            c = cost.get_cost(i, 1, 1)
            expected = -gain_for_week_control_and_scenario(residual_load, i, 1, 1, 6)
            assert c <= expected, "week: " + str(i) + " cost : " + str(c)

        elif res.get_previous_monday(7*i) + 7 < res.first_day or res.get_previous_monday(7*i) >= res.last_day:
            c = cost.get_cost(i, 0, 1)
            assert c == 0, "week: " + str(i) + " cost : " + str(c)
            c = cost.get_cost(i, 1, 1)
            assert c == 0, "week: " + str(i) + " cost : " + str(c)
        else:
            c = cost.get_cost(i, 0, 3)
            expected = -gain_for_week_control_and_scenario(residual_load, i, 3, 0, 6)
            assert c == pytest.approx(expected), "week: " + str(i) + " cost : " + str(c)

            c = cost.get_cost(i, 1, 1)
            expected = -gain_for_week_control_and_scenario(residual_load, i, 1, 1, 6)
            assert c == pytest.approx(expected), "week: " + str(i) + " cost : " + str(c)

            assert cost.get_cost(i, 1, 0) == 0


def test_different_week_day_white():
    for d in range(7):
        nb_sce = 1
        residual_load = np.random.rand(constants.NB_DAYS + 1, nb_sce)*1000
        res = TempoReservoir(first_day=0, last_day=constants.NB_DAYS)
        res.week_day_first_september = d
        cost = TempoCostFunction(residual_load, res)

        for i in range(constants.RESULTS_SIZE):
            if res.get_previous_monday(7*i+(7-d)%7) < res.first_day or res.get_previous_monday(7*i+(7-d)%7) > res.last_day:
                c = cost.get_cost(i, 0, 1)
                assert c == 0, "week: " + str(i) + " cost : " + str(c) + " week_day 1st september : " + str(d)
            else:
                c = cost.get_cost(i, 0, 3)
                assert c != 0, "week: " + str(i) + " week_day 1st september : " + str(d)
