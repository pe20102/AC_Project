Accelerated Computing Mini Project - Accelerating Metropolis Monte Carlo Simulations.

The directories contain the following code files:
Base: Base code given for the task, completely unchanged.
Numpy: LebwohlLasher_NPY.py = The simulation for the Numpy Vectorisation method.
Numba: LebwohlLasher_NB.py = The simulation for the Numba implementation.
Cython: lebwohl_lasher_CYTH.pyx = The Cython file for the Cython implementation. LebwohlLasher_CMULT.pyx = The Cython file which includes Multithreading. This directory also contains the appropriate setup and run files.
MPI: LebwohlLasher_MPI.py = The simulation for the MPI implementation.
