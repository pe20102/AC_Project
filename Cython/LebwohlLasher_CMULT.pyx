"""
Cython-optimized Lebwohl-Lasher simulation with multi-threading support
"""

# Import statements
import cython
from cython.parallel import prange
from openmp cimport omp_get_max_threads, omp_set_num_threads
cimport numpy as np
import numpy as np
from libc.math cimport cos, sin, exp, pi
from libc.stdlib cimport rand, RAND_MAX
import sys
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add compiler directives at the module level
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

#=======================================================================
def initdat(int nmax):
    """Initialize lattice with random orientations"""
    return np.random.random_sample((nmax,nmax))*2.0*np.pi

#=======================================================================
def plotdat(np.ndarray[np.float64_t, ndim=2] arr, int pflag, int nmax):
    """Plot the lattice configuration"""
    if pflag==0:
        return
        
    cdef np.ndarray[np.float64_t, ndim=2] u = np.cos(arr)
    cdef np.ndarray[np.float64_t, ndim=2] v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cdef np.ndarray[np.float64_t, ndim=2] cols = np.zeros((nmax,nmax))
    
    if pflag==1:
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2:
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()

#=======================================================================
cdef double one_energy(double[:, :] arr, int ix, int iy, int nmax) nogil:
    """Compute energy of a single cell"""
    cdef double en = 0.0
    cdef int ixp = (ix+1)%nmax
    cdef int ixm = (ix-1)%nmax
    cdef int iyp = (iy+1)%nmax
    cdef int iym = (iy-1)%nmax
    cdef double ang

    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    return en

#=======================================================================
cdef double all_energy(double[:, :] arr, int nmax):
    """Compute energy of entire lattice"""
    cdef double enall = 0.0
    cdef int i, j
    
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall

#=======================================================================
def get_order(double[:, :] arr, int nmax):
    """Calculate order parameter using Q tensor approach"""
    cdef np.ndarray[np.float64_t, ndim=2] Qab = np.zeros((3,3))
    cdef np.ndarray[np.float64_t, ndim=2] delta = np.eye(3,3)
    cdef int a, b, i, j
    
    # Convert memory view to numpy array for calculations
    cdef np.ndarray[np.float64_t, ndim=2] arr_np = np.asarray(arr)
    
    # Generate 3D unit vectors
    cdef np.ndarray[np.float64_t, ndim=3] lab = np.vstack((
        np.cos(arr_np),
        np.sin(arr_np),
        np.zeros_like(arr_np)
    )).reshape(3,nmax,nmax)
    
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    
    Qab = np.array(Qab)/(2.0*nmax*nmax)
    eigenvalues = np.linalg.eigvals(Qab)
    return np.max(eigenvalues.real)

#=======================================================================
cdef double MC_step_parallel(double[:, :] arr, double Ts, int nmax, int num_threads) nogil:
    """Perform one Monte Carlo step using multiple threads"""
    cdef double scale = 0.1 + Ts
    cdef int accept = 0
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz
    
    # Set number of OpenMP threads
    omp_set_num_threads(num_threads)
    
    for i in prange(nmax, nogil=True, schedule='guided', num_threads=num_threads):
        for j in range(nmax):
            ix = i
            iy = j
            ang = scale * (2.0 * rand() / RAND_MAX - 1.0)
            
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            
            if en1 <= en0:
                accept += 1
            else:
                boltz = exp(-(en1 - en0) / Ts)
                if boltz >= rand() / RAND_MAX:
                    accept += 1
                else:
                    arr[ix,iy] -= ang
                    
    return accept/(nmax*nmax)

#=======================================================================
def main(str program, int nsteps, int nmax, double temp, int pflag, int num_threads):
    """Main simulation function"""
    # Initialize arrays with both numpy arrays and memory views
    cdef np.ndarray[np.float64_t, ndim=2] lattice_np = initdat(nmax)
    cdef double[:, :] lattice = lattice_np
    cdef np.ndarray[np.float64_t, ndim=1] energy_np = np.zeros(nsteps+1)
    cdef np.ndarray[np.float64_t, ndim=1] ratio_np = np.zeros(nsteps+1)
    cdef np.ndarray[np.float64_t, ndim=1] order_np = np.zeros(nsteps+1)
    cdef double[:] energy = energy_np
    cdef double[:] ratio = ratio_np
    cdef double[:] order = order_np
    cdef int it
    cdef double runtime
    
    plotdat(lattice_np, pflag, nmax)
    
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)

    initial = time.time()
    for it in range(1, nsteps+1):
        ratio[it] = MC_step_parallel(lattice, temp, nmax, num_threads)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    runtime = time.time() - initial
    
    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}, "
          f"Threads: {num_threads}, Order: {order[nsteps-1]:5.3f}, "
          f"Time: {runtime:8.6f} s")
    
    savedat(lattice_np, nsteps, temp, runtime, ratio_np, energy_np, order_np, nmax)
    plotdat(lattice_np, pflag, nmax)

#=======================================================================
def savedat(np.ndarray[np.float64_t, ndim=2] arr, int nsteps, double Ts, 
            double runtime, np.ndarray[np.float64_t, ndim=1] ratio, 
            np.ndarray[np.float64_t, ndim=1] energy, 
            np.ndarray[np.float64_t, ndim=1] order, int nmax):
    """Save simulation data to file"""
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"LL-Output-{current_datetime}.txt"
    
    with open(filename, "w") as FileOut:
        print("#=====================================================", file=FileOut)
        print(f"# File created:        {current_datetime}", file=FileOut)
        print(f"# Size of lattice:     {nmax}x{nmax}", file=FileOut)
        print(f"# Number of MC steps:  {nsteps}", file=FileOut)
        print(f"# Reduced temperature: {Ts:5.3f}", file=FileOut)
        print(f"# Run time (s):        {runtime:8.6f}", file=FileOut)
        print("#=====================================================", file=FileOut)
        print("# MC step:  Ratio:     Energy:   Order:", file=FileOut)
        print("#=====================================================", file=FileOut)
        
        for i in range(nsteps+1):
            print(f"   {i:05d}    {ratio[i]:6.4f} {energy[i]:12.4f}  {order[i]:6.4f} ", 
                  file=FileOut)

#=======================================================================
if __name__ == '__main__':
    if len(sys.argv) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        THREADS = int(sys.argv[5])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, THREADS)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>")