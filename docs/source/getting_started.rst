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


Using the command-line
----------------------

The command line works with a yaml setting file:

.. code-block::

    python .\proxy_bellman\src\proxy_bellman.py yaml-settings path-to/settings.yml

To use the hydro algorithm, the yaml file must contain:

+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| field                  | Type            | Optional | Default value | Description                                                                          |
+========================+=================+==========+===============+======================================================================================+
| hydro                  |                 | No       |               |Main field containing all settings.                                                   |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| study                  | Path            | Yes      | (see desc)    | Path to the study on which to run the algorithm. Mandatory if no host field.         |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| host                   | string          | Yes      | (see desc)    | Host for antares web server. Mandatory if no study field.                            |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| token                  | string          | Yes      | (see desc)    | Token for antares web server. Usefull (but optional) only with host field.           |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| study_id               | string          | Yes      | (see desc)    | Study id for antares web server. Mandatory with host field.                          |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| areas                  | list(string)    | No       |               | List of areas two compute. Each item must contain the name of one area of the study. |
|                        |                 |          |               | Must contain at least one item.                                                      |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| output_dir             | Path            | Yes      | "./results"   | Path to the results folder. Will be appended by the date and time.                   |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| tmp_dir                | Path            | Yes      | "./tmp"       | Path to the folder where to put back up files for modify or undo actions.            |
|                        |                 |          |               | "./tmp" if not provided.                                                             |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| mc_years               | int or interval | Yes      | 200           | List of years for which to compute trajectories.                                     |
|                        | or list(int)    |          |               |                                                                                      |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| ts_selection           | int or interval | Yes      | None          | List of years to take into account for computing bellman values.                     |
|                        | or list(int)    |          |               |                                                                                      |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| nb_turb                | int             | Yes      | 25            | Number of values on which to compute the cost function.                              |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| alpha                  | int             | Yes      | 2             | Parameter for the computation of the costs value and the turbine vs pumping ratio.   |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| penalty_factor         | float           | Yes      | 1             | Factor to modulate how important it is to respect guidelines.                        |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| actions                | list(string)    | Yes      | None          | Actions to perform. Each item must contain an action. Must contain at least one item.|
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+

Possible actions are :

* export_controls: Writes a csv file with all computed controls.
* export_trajectories: Writes a csv file with all computed trajectories.
* modify_antares_data: Applies controls to provided study.
* undo_study: After using modify_antares_data, returns the study to its original state.

To use the tempo algorithm, the yaml file must contain:

+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| field                  | Type            | Optional | Default value | Description                                                                          |
+========================+=================+==========+===============+======================================================================================+
| tempo                  |                 | No       |               | Main field containing all settings.                                                  |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| study                  | Path            | Yes      | (see desc)    | Path to the study on which to run the algorithm. Mandatory if no host field.         |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| host                   | string          | Yes      | (see desc)    | Host for antares web server. Mandatory if no study field.                            |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| token                  | string          | Yes      | (see desc)    | token for antares web server. Usefull (but optional) only with host field.           |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| study_id               | string          | Yes      | (see desc)    | Study id for antares web server. Mandatory with host field.                          |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| areas                  | list(string)    | No       |               | List of areas two compute. Each item must contain the name of one area of the study. |
|                        |                 |          |               | Must contain at least one item.                                                      |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| output_dir             | Path            | Yes      | "./results"   | Path to the results folder. Will be appended by the date and time.                   |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| mc_years               | int or interval | Yes      | 200           | List of years for which to compute trajectories.                                     |
|                        | or list(int)    |          |               |                                                                                      |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| ts_selection           | int or interval | Yes      | None          | List of years to take into account for computing bellman values.                     |
|                        | or list(int)    |          |               |                                                                                      |
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+
| actions                | list(string)    | Yes      | None          | Actions to perform. Each item must contain an action. Must contain at least one item.|
+------------------------+-----------------+----------+---------------+--------------------------------------------------------------------------------------+

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