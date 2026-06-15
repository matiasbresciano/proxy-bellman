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

You can also run the proxy from the command line with :

.. code-block::

    python .\proxy_bellman\src\proxy_bellman.py hydro path_to_study area --actions=action

More infos with :

.. code-block::

    python .\proxy_bellman\src\proxy_bellman.py hydro --help

or :

.. code-block::

    python .\proxy_bellman\src\proxy_bellman.py tempo --help