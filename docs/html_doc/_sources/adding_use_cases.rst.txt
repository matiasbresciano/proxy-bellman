Adding use cases
================

After installing the requirements as indicated in getting started, install dev requirement in the same manner:

.. code-block::

    python -m pip install -r proxy-bellman/requirements-dev.txt

You can run the tests to ensure everything works:

.. code-block::

    cd proxy_bellman

    pytest .\tests

To create a new use case create a new directory in src: "my_use_case". In this directory,
you will need to create the files for bellman, cost_function, proxy, reservoir, trajectory, in
which you will create derivatives for all the classes in the base package such as "MyUseCaseBellman"
(derivative from base.Bellman), "MyUseCaseCostFunction" (derivative from base.CostFunction), etc. Every
abstract function of these classes need to be overridden, other functions can be overridden if needed.

You should add relevant tests for your code in the folder "tests".

You can use mypy to check type correctness with:

.. code-block::

    python -m mypy proxy_bellman/src

You can generate html documentation for your code by creating a new rst file "docs/source", "my_use_case.rst", which can be
copied and adjusted from "hydro.rst" or "tempo.rst". Link your newly created file in "modules.rst" bellow "tempo".
Finally, generate the html with the command:

.. code-block::

    .\proxy_bellman\docs\make.bat html
