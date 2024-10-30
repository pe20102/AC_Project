"""
Basic Python Lebwohl-Lasher code with vectorization for improved performance.
Based on P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#=======================================================================
def initdat(nmax):
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    return arr

#=======================================================================
def plotdat(arr, pflag, nmax):
    if pflag == 0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax, nmax))
    if pflag == 1:  # color the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i, j] = one_energy_vectorized(arr, nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:  # color the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()

#=======================================================================
def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    with open(filename, "w") as FileOut:
        # Write a header with run parameters
        print("#=====================================================", file=FileOut)
        print("# File created:        {:s}".format(current_datetime), file=FileOut)
        print("# Size of lattice:     {:d}x{:d}".format(nmax, nmax), file=FileOut)
        print("# Number of MC steps:  {:d}".format(nsteps), file=FileOut)
        print("# Reduced temperature: {:5.3f}".format(Ts), file=FileOut)
        print("# Run time (s):        {:8.6f}".format(runtime), file=FileOut)
        print("#=====================================================", file=FileOut)
        print("# MC step:  Ratio:     Energy:   Order:", file=FileOut)
        print("#=====================================================", file=FileOut)
        # Write the columns of data
        for i in range(nsteps + 1):
            print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i, ratio[i], energy[i], order[i]), file=FileOut)

#=======================================================================
def one_energy_vectorized(arr, nmax):
    """
    Vectorized computation of the energy for the entire lattice.
    """
    # Using periodic boundary conditions by rolling arrays
    en_x = 0.5 * (1.0 - 3.0 * np.cos(arr - np.roll(arr, shift=1, axis=0)) ** 2)
    en_y = 0.5 * (1.0 - 3.0 * np.cos(arr - np.roll(arr, shift=1, axis=1)) ** 2)
    
    # Summing energy contributions for all pairs
    total_energy = np.sum(en_x + en_y)
    return total_energy

def all_energy_vectorized(arr, nmax):
    """
    Computes the total energy for the lattice using the vectorized one_energy function.
    """
    return one_energy_vectorized(arr, nmax)

#=======================================================================
def get_order(arr, nmax):
    Qab = np.zeros((3, 3))
    delta = np.eye(3, 3)

    lab = np.vstack((np.cos(arr), np.sin(arr), np.zeros_like(arr))).reshape(3, nmax, nmax)
    for a in range(3):
        for b in range(3):
            Qab[a, b] += np.sum(3 * lab[a] * lab[b] - delta[a, b])
    Qab = Qab / (2 * nmax * nmax)
    eigenvalues, _ = np.linalg.eig(Qab)
    return eigenvalues.max()

#=======================================================================
def MC_step_vectorized(arr, Ts, nmax):
    """
    Vectorized version of the Monte Carlo step.
    """
    scale = 0.1 + Ts
    accept = 0

    # Generate all random indices and angle changes at once
    xran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    yran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    aran = np.random.normal(0.0, scale, (nmax, nmax))
    
    # Calculate initial energy for all random positions
    en0 = one_energy_vectorized(arr, nmax)
    
    # Perform Monte Carlo step in a vectorized way
    arr[xran, yran] += aran
    en1 = one_energy_vectorized(arr, nmax)
    
    # Determine acceptance based on the Metropolis criterion
    accept_mask = (en1 <= en0) | (np.exp(-(en1 - en0) / Ts) >= np.random.uniform(0.0, 1.0, size=(nmax, nmax)))
    arr[xran, yran] -= aran * (~accept_mask)
    accept = np.mean(accept_mask)

    return accept

#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    lattice = initdat(nmax)
    plotdat(lattice, pflag, nmax)
    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio = np.zeros(nsteps + 1, dtype=np.float64)
    order = np.zeros(nsteps + 1, dtype=np.float64)

    energy[0] = all_energy_vectorized(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)

    initial = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step_vectorized(lattice, temp, nmax)
        energy[it] = all_energy_vectorized(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    final = time.time()
    runtime = final - initial

    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:.3f}, "
          f"Order: {order[nsteps - 1]:.3f}, Time: {runtime:.6f} s")

    plotdat(lattice, pflag, nmax)
    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)

#=======================================================================
if __name__ == '__main__':
    if len(sys.argv) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
