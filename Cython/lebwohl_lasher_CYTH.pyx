"""
Cython-optimized Lebwohl-Lasher simulation
"""

# Import statements
import cython
cimport numpy as np
import numpy as np
from libc.math cimport cos, sin, exp, pi
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
cdef double one_energy(np.ndarray[np.float64_t, ndim=2] arr, int ix, int iy, int nmax):
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
cdef double all_energy(np.ndarray[np.float64_t, ndim=2] arr, int nmax):
    """Compute energy of entire lattice"""
    cdef double enall = 0.0
    cdef int i, j
    
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall

#=======================================================================
def get_order(np.ndarray[np.float64_t, ndim=2] arr, int nmax):
    """Calculate order parameter using Q tensor approach"""
    cdef np.ndarray[np.float64_t, ndim=2] Qab = np.zeros((3,3))
    cdef np.ndarray[np.float64_t, ndim=2] delta = np.eye(3,3)
    cdef int a, b, i, j
    
    # Generate 3D unit vectors
    cdef np.ndarray[np.float64_t, ndim=3] lab = np.vstack((
        np.cos(arr),
        np.sin(arr),
        np.zeros_like(arr)
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
cdef double MC_step(np.ndarray[np.float64_t, ndim=2] arr, double Ts, int nmax):
    """Perform one Monte Carlo step"""
    cdef double scale = 0.1 + Ts
    cdef int accept = 0
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz
    
    cdef np.ndarray[np.float64_t, ndim=2] xran = np.random.randint(0, high=nmax, size=(nmax,nmax)).astype(np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] yran = np.random.randint(0, high=nmax, size=(nmax,nmax)).astype(np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] aran = np.random.normal(scale=scale, size=(nmax,nmax))

    for i in range(nmax):
        for j in range(nmax):
            ix = int(xran[i,j])
            iy = int(yran[i,j])
            ang = aran[i,j]
            
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            
            if en1 <= en0:
                accept += 1
            else:
                boltz = exp(-(en1 - en0) / Ts)
                if boltz >= np.random.uniform(0.0, 1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang
                    
    return accept/(nmax*nmax)

#=======================================================================
def main(str program, int nsteps, int nmax, double temp, int pflag):
    """Main simulation function"""
    cdef np.ndarray[np.float64_t, ndim=2] lattice = initdat(nmax)
    cdef np.ndarray[np.float64_t, ndim=1] energy = np.zeros(nsteps+1)
    cdef np.ndarray[np.float64_t, ndim=1] ratio = np.zeros(nsteps+1)
    cdef np.ndarray[np.float64_t, ndim=1] order = np.zeros(nsteps+1)
    cdef int it
    cdef double runtime
    
    plotdat(lattice, pflag, nmax)
    
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)

    initial = time.time()
    for it in range(1, nsteps+1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    runtime = time.time() - initial
    
    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}: "
          f"Order: {order[nsteps-1]:5.3f}, Time: {runtime:8.6f} s")
    
    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)

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
    if len(sys.argv) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")