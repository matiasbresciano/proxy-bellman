tempo package
=============

How the tempo algorithm handles input year
------------------------------------------

The tempo contract works on a year running from September the first of year N to August the thirty-first of year N+1.
September the first is the date at which the tempo days stocks are replenished, and for this reason, the tempo algorithm
also needs to work on a year running from September to August. To be able to process input years starting at any month,
the following preprocess is done by the app:

1. Remove the last day of the input year (last two days for leap year)

2.

    * If the first month of the input year is between May and August, cut it
      between the August 31st and September 1st then put the first half of the input year (from starting month to
      August 31st) after the second half of the input year. As a result, the first half of the input year gets shifted
      by one day (for example, August 31st becomes August 30). The following figure shows this process on an example
      with an input year starting in July.

      .. image:: preprocess_july.png

     * If the first month of the input year is between October and Marsh, cut it between 30 August and 31st August, then
       put the first half of the input year (from starting month to August 30) after the second half of the input year.
       The second half of the input year gets shifted by one day (for example, August 31st becomes September 1st). The
       following figure shows this process on an example with an input year starting in January.

      .. image:: preprocess_january.png

     * If the first month of the input year is April, the preprocess is the same as in the previous case, except that
       the first day of the input year is removed in place of the last.

     * If the first month of the input year is September, the last day of the input year (August 31st) won't be
       taken into account, but no preprocessing is necessary.


After the computations, controls and trajectories are returned in the same order as the input year, only one day (in
most cases, the last day of the input year) as not been taken into account and must be considered as a blue tempo day
to keep coherence in the stock values.

This preprocessing technique as been designed to keep Marsh data as exact as possible. The 31st of Marsh is the last
day where it is possible to have a red tempo day in a year, and it has been observed that a lot of red tempo days are
used during this month. Keeping Marsh data unshifted prevents incoherences such as having a red tempo day on the 1st of
April or a blue tempo day on the 31st of Marsh that should have been red.

Submodules
----------

tempo.cost\_function module
---------------------------

.. automodule:: tempo.cost_function
   :members:
   :show-inheritance:
   :undoc-members:

tempo.proxy module
------------------

.. automodule:: tempo.proxy
   :members:
   :show-inheritance:
   :undoc-members:

tempo.reservoir module
----------------------

.. automodule:: tempo.reservoir
   :members:
   :show-inheritance:

tempo.trajectory module
-----------------------

.. automodule:: tempo.trajectory
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tempo
   :members:
   :show-inheritance:
   :undoc-members:
