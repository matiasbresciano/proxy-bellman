import pytest
import numpy as np

from tempo.proxy import TempoAntaresProxy
from tempo.reservoir import TempoReservoir

dir_study = "test_data/two_nodes"
area = "area1"

proxy = TempoAntaresProxy(dir_study, area, 10)
bellman_values_red = proxy.get_bellman_values()[0]
trajectories_red = proxy.get_trajectories()[0]
controls_red = proxy.get_controls()[0]

bellman_values_white_and_red = proxy.get_bellman_values()[1]
trajectories_white_and_red = proxy.get_trajectories()[1]
controls_white_and_red = proxy.get_controls()[1]

gain_function_red = proxy._proxy._cost_function[0]
gain_function_white_and_red = proxy._proxy._cost_function[1]

red_reservoir = proxy._proxy._reservoir[0]


def test_gain_function_tempo() -> None:
    assert isinstance(red_reservoir, TempoReservoir)
    cost = gain_function_red.get_cost(10, 0, 5)
    assert cost == pytest.approx(-4559276.533)
    cost = gain_function_red.get_cost(20, 5, 3)
    assert cost == pytest.approx(-2493107.2824999997)
    cost = gain_function_red.get_cost(22, 9, 0)
    assert cost == pytest.approx(0)
    cost = gain_function_white_and_red.get_cost(2, 0, 6)
    assert cost == pytest.approx(-4480186.672499999)


def test_bellman_values_tempo() -> None:
    assert bellman_values_red.shape == (53, 23)
    assert bellman_values_red[17, 10] == pytest.approx(14886250.286077848)
    assert bellman_values_red[30] == pytest.approx(np.zeros(23))
    assert bellman_values_red[10] == pytest.approx(np.array([
            0,
            1635463.0805157232,
            3228041.4476605295,
            4791119.439975521,
            6329318.290087402,
            7848022.742237193,
            9343043.539659884,
            10814433.293034298,
            12269016.144185368,
            13705500.936216895,
            15123858.057936007,
            16524442.690887073,
            17911563.972539164,
            19284557.55051198,
            20643440.72810308,
            21988573.63809376,
            23319167.41347275,
            24636573.917794164,
            25942181.54194791,
            27236704.665245306,
            28520379.91386708,
            29792372.27432084,
            31052861.257673346]
            ))


def test_optimal_trajectories() -> None:
    assert trajectories_red.shape == (10, 52)
    assert controls_red.shape == (10, 52)
    assert not np.any(trajectories_red[0, :10]-22)
    expected_stock_trajectory = np.array([
            22.0,     
            21.0,  
            17.0,   
            14.0,     
            14.0,     
            14.0,     
            11.0,     
            9.0 ,     
            5.0 ,     
            5.0 ,     
            2.0 ,     
            1.0 ,     
            1.0 ,     
            1.0 ,
            0,
            0,
            0,
            0,
            0,
            0,
            0
    ])
    assert not np.any(trajectories_red[0, 10:31] - expected_stock_trajectory)
    # assert trajectories_red[0] ==pytest.approx(
    #     trajectories_red.stock_trajectory_for_scenario(0)
    # ) bizarre je sais pas ce que c'est censé tester
    assert not np.any(trajectories_red[0, 31:] - np.repeat(0, 21))

    expected_control_trajectory = np.array([
            0.0,
            1.0,
            4.0,
            3.0,
            0.0,
            0.0,
            3.0,
            2.0,
            4.0,
            0.0,
            3.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0
    ])
    assert not np.any(controls_red[0, 10:31] - expected_control_trajectory)
    # assert controls_red[0] == pytest.approx(
    #     trajectories_red.control_trajectory_for_scenario(0)
    # )
    assert not np.any(controls_red[0, :10])
    assert not np.any(controls_red[0, 31:])


def test_white_and_red_trajectories() -> None:
    expected_white_trajectory = np.array([
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            40.0     	,
            39.0     	,
            39.0     	,
            36.0     	,
            35.0     	,
            34.0     	,
            31.0     	,
            30.0     	,
            27.0     	,
            23.0     	,
            23.0     	,
            21.0     	,
            19.0     	,
            17.0     	,
            15.0     	,
            12.0     	,
            8.0      	,
            8.0      	,
            3.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0
    ])
    trajectories_white = trajectories_white_and_red - trajectories_red
    assert np.all(trajectories_white_and_red >= trajectories_red)
    diff_traj = trajectories_white[0] - expected_white_trajectory
    assert not np.any(diff_traj)
