Getting started
===============

Installing the environment
--------------------------

Create a new directory and place the project proxy_bellman inside.

Create and activate a venv in this directory and install requirements:

.. code-block::

    python -m venv .venv

    source .venv/bin/activate

    python -m pip install -r proxy-bellman/requirements.txt

In the same directory, create a new file "file.py". Inside you can write your script. For example:

.. code-block:: python

    import proxy_bellman.src.hydro.HydroAntaresProxy

    dir_study= "proxy-bellman/test_data/two_nodes"

    area1 = "area1"

    proxy = HydroAntaresProxy(dir_study, area1, np.arange(10), alpha=2, penalty_factor=0.4)

    proxy.export_trajectories()

The command line works with a yaml setting file:

.. code-block::

    python .\proxy_bellman\src\proxy_bellman.py yaml-settings path-to/settings.yml

To use the hydro algorithm, the yaml file must contain:

+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| field                  | Type            | Description                                                                          | Optional |
+========================+=================+======================================================================================+==========+
| hydro                  |                 | Main field containing all settings.                                                  | No       |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| study                  | Path            | Path to the study on which to run the algorithm.                                     | No       |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| areas                  | list(string)    | List of areas two compute. Each item must contain the name of one area of the study. | No       |
|                        |                 | Must contain at least one item.                                                      |          |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| output_dir             | Path            | Path to the results folder. Will be appended by the date and time.                   | No       |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| mc_years               | int or interval | List of years for which to compute trajectories.                                     | No       |
|                        | or list(int)    |                                                                                      |          |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| ts_selection           | int or interval | List of years to take into account for computing bellman values.                     | No       |
|                        | or list(int)    |                                                                                      |          |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| nb_turb                | int             | Number of values on which to compute the cost function.                              | No       |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| alpha                  | int             | Parameter for the computation of the costs value and the turbine vs pumping ratio.   | No       |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| penalty_factor         | float           | Factor to modulate how important it is to respect guidelines.                        | No       |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| actions                | list(string)    | Actions to perform. Each item must contain an action. Must contain at least one item.| Yes      |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+

Possible actions are :

* export_controls: Writes a csv file with all computed controls.
* export_trajectories: Writes a csv file with all computed trajectories.
* modify_antares_data: Applies controls to provided study.
* undo_study: After using modify_antares_data, returns the study to its original state.

To use the tempo algorithm, the yaml file must contain:

+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| field                  | Type            | Description                                                                          | Optional |
+========================+=================+======================================================================================+==========+
| tempo                  |                 | Main field containing all settings.                                                  | No       |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| study                  | Path            | Path to the study on which to run the algorithm.                                     | No       |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| areas                  | list(string)    | List of areas two compute. Each item must contain the name of one area of the study. | No       |
|                        |                 | Must contain at least one item.                                                      |          |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| output_dir             | Path            | Path to the results folder. Will be appended by the date and time.                   | No       |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| mc_years               | int or interval | List of years for which to compute trajectories.                                     | No       |
|                        | or list(int)    |                                                                                      |          |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| ts_selection           | int or interval | List of years to take into account for computing bellman values.                     | No       |
|                        | or list(int)    |                                                                                      |          |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+
| actions                | list(string)    | Actions to perform. Each item must contain an action. Must contain at least one item.| Yes      |
+------------------------+-----------------+--------------------------------------------------------------------------------------+----------+

Possible actions are :

* export_controls: Writes a csv file with all computed controls.
* export_trajectories: Writes a csv file with all computed trajectories.
* export_calendar: Writes a csv file for each mc year intended to be used by D3E algorithm.

You can also run the proxy from the command line with :

.. code-block::

    python .\proxy_bellman\src\proxy_bellman.py hydro path_to_study area --actions=action

More infos with :

.. code-block::

    python .\proxy_bellman\src\proxy_bellman.py hydro --help

or :

.. code-block::

    python .\proxy_bellman\src\proxy_bellman.py tempo --help