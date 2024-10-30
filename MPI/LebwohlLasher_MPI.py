import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI

MAXWORKER = 3          # maximum number of worker tasks
MINWORKER = 1          # minimum number of worker tasks
BEGIN = 1              # message tag
DONE = 2               # message tag
ATAG = 3              # message tag
BTAG = 4              # message tag
NONE = 0              # indicates no neighbour
MASTER = 0            # taskid of first process

def initdat(lattice_size):
    """
    Arguments:
      lattice_size (int) = size of lattice to create (lattice_size,lattice_size).
    Description:
      Function to create and initialise the main data array that holds
      the lattice. Will return a square lattice (size lattice_size x lattice_size)
      initialised with random orientations in the range [0,2pi].
    Returns:
      lattice (float(lattice_size,lattice_size)) = array to hold lattice.
    """
    lattice = np.random.random_sample((lattice_size,lattice_size))*2.0*np.pi
    return lattice

def plotdat(angles, energies, plot_flag, lattice_size):
    """
    Arguments:
      angles (float(lattice_size,lattice_size)) = array that contains lattice angles;
      energies (float(lattice_size,lattice_size)) = array that contains lattice energies;
      plot_flag (int) = parameter to control plotting;
      lattice_size (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array. Makes use of the
      quiver plot style in matplotlib. Use plot_flag to control style:
        plot_flag = 0 for no plot (for scripted operation);
        plot_flag = 1 for energy plot;
        plot_flag = 2 for angles plot;
        plot_flag = 3 for black plot.
    Returns:
      NULL
    """
    if plot_flag == 0:
        return
    u = np.cos(angles)
    v = np.sin(angles)
    x = np.arange(lattice_size)
    y = np.arange(lattice_size)
    if plot_flag == 1:  # colour the arrows according to energy
        rc('image', cmap='rainbow')
        cols = energies
        norm = plt.Normalize(cols.min(), cols.max())
    elif plot_flag == 2:  # colour the arrows according to angle
        rc('image', cmap='hsv')
        cols = angles % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        rc('image', cmap='gist_gray')
        cols = np.zeros_like(angles)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*lattice_size)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()

def savedat(arr, num_steps, temperature, runtime, acceptance_ratio, total_energy, order, lattice_size):
    """
    Arguments:
      arr (float(lattice_size,lattice_size)) = array that contains lattice data;
      num_steps (int) = number of Monte Carlo steps (MCS) performed;
      temperature (float) = reduced temperature (range 0 to 2);
      acceptance_ratio (float(num_steps)) = array of acceptance ratios per MCS;
      total_energy (float(num_steps)) = array of reduced energies per MCS;
      order (float(num_steps)) = array of order parameters per MCS;
      lattice_size (int) = side length of square lattice to simulated.
    """
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(lattice_size,lattice_size),file=FileOut)
    print("# Number of MC steps:  {:d}".format(num_steps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(temperature),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    for i in range(num_steps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(
            i,acceptance_ratio[i],total_energy[i],order[i]),file=FileOut)
    FileOut.close()

def block_energy(num_rows, angles, energies, parity):
    """Calculate block energy for the lattice"""
    for ix in range(1, num_rows+1):
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos(angles[ix,parity::2]-angles[ix+1,parity::2]))**2
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos(angles[ix,parity::2]-angles[ix-1,parity::2]))**2
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos(angles[ix,parity::2]-np.roll(angles[ix,:],-1)[(parity)::2]))**2
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos(angles[ix,parity::2]-np.roll(angles[ix,:],1)[(parity)::2]))**2
        parity = 1 - parity
    parity = (parity + num_rows%2)%2

def partial_Q(angles, lattice_size):
    """Calculate partial order tensor"""
    (dimx, dimy) = np.shape(angles)
    field = np.vstack((np.cos(angles),np.sin(angles))).reshape((2,dimx,dimy))
    order_tensor = 1.5*np.einsum("aij,bij->ab",field,field)/(lattice_size**2) - 0.5*(dimx*dimy/(lattice_size**2))*np.eye(2,2)
    return order_tensor

def get_order(order_tensor, num_steps):
    """Calculate order parameter from order tensor"""
    order = np.zeros(num_steps)
    for t in range(num_steps):
        eigenvalues, eigenvectors = np.linalg.eig(order_tensor[t])
        order[t] = eigenvalues.max()
    return order

def MC_substep(comm, angles, random_angles, energies, temperature, lattice_size, num_rows, 
               parity, above_rank, below_rank, acceptance_ratio, step):
    """Perform a Monte Carlo substep"""
    
    # Communicate border rows with neighbours
    req = comm.Isend([angles[1,:], lattice_size, MPI.DOUBLE], dest=above_rank, tag=ATAG)
    req = comm.Isend([angles[num_rows,:], lattice_size, MPI.DOUBLE], dest=below_rank, tag=BTAG)
    comm.Recv([angles[0,:], lattice_size, MPI.DOUBLE], source=above_rank, tag=BTAG)
    comm.Recv([angles[num_rows+1,:], lattice_size, MPI.DOUBLE], source=below_rank, tag=ATAG)
    block_energy(num_rows, angles, energies[0], parity)

    # Perturb the angles randomly (for on-parity sites)
    if parity == 0:
        angles[(1+parity):num_rows+1:2,parity::2] += random_angles[parity::2,parity::2]
        angles[(2-parity):num_rows+1:2,(1-parity)::2] += random_angles[(1-parity)::2,(1-parity)::2]
    else:
        angles[(1+parity):num_rows+1:2,(1-parity)::2] += random_angles[parity::2,(1-parity)::2]
        angles[(2-parity):num_rows+1:2,parity::2] += random_angles[(1-parity)::2,parity::2]

    # Communicate new border rows with neighbours
    req = comm.Isend([angles[1,:], lattice_size, MPI.DOUBLE], dest=above_rank, tag=ATAG)
    req = comm.Isend([angles[num_rows,:], lattice_size, MPI.DOUBLE], dest=below_rank, tag=BTAG)
    comm.Recv([angles[0,:], lattice_size, MPI.DOUBLE], source=above_rank, tag=BTAG)
    comm.Recv([angles[num_rows+1,:], lattice_size, MPI.DOUBLE], source=below_rank, tag=ATAG)

    # Compute new energies
    block_energy(num_rows, angles, energies[1], parity)

    # Determine which changes to accept
    guaranteed = (energies[1] <= energies[0])
    boltz = np.exp(-(energies[1]-energies[0])/temperature)
    accept = guaranteed + (1-guaranteed)*(boltz >= np.random.uniform(0.0,1.0,size=(num_rows,lattice_size)))

    # Adjust energies based on which sites were accepted
    energies[1] = accept*energies[1] + (1-accept)*energies[0]

    if parity == 0:
        # Record this rank's portion of the total acceptance ratio
        acceptance_ratio[step] += (np.sum(accept[parity::2,parity::2])+
                                 np.sum(accept[1-parity::2,1-parity::2]))/(lattice_size**2)

        # Undo the rejected changes
        angles[(1+parity):num_rows+1:2,parity::2] -= (1-accept[parity::2,parity::2])*random_angles[parity::2,parity::2]
        angles[(2-parity):num_rows+1:2,(1-parity)::2] -= (1-accept[(1-parity)::2,(1-parity)::2])*random_angles[(1-parity)::2,(1-parity)::2]
    else:
        # Record this rank's portion of the total acceptance ratio
        acceptance_ratio[step] += (np.sum(accept[parity::2,(1-parity)::2])+
                                 np.sum(accept[(1-parity)::2,parity::2]))/(lattice_size**2)

        # Undo the rejected changes
        angles[(1+parity):num_rows+1:2,(1-parity)::2] -= (1-accept[parity::2,(1-parity)::2])*random_angles[parity::2,(1-parity)::2]
        angles[(2-parity):num_rows+1:2,parity::2] -= (1-accept[(1-parity)::2,parity::2])*random_angles[(1-parity)::2,parity::2]

def MC_step(comm, angles, energies, temperature, lattice_size, num_rows, offset, 
            above_rank, below_rank, total_energy, order_tensor, acceptance_ratio, step):
    """Perform a full Monte Carlo step"""
    
    scale = 0.1 + temperature
    random_angles = np.random.normal(scale=scale, size=(num_rows, lattice_size))
    energies = np.zeros((2, num_rows, lattice_size))

    # First substep
    parity = offset % 2
    MC_substep(comm, angles, random_angles, energies, temperature, lattice_size, 
               num_rows, parity, above_rank, below_rank, acceptance_ratio, step)

    # Second substep
    parity = 1 - offset % 2
    MC_substep(comm, angles, random_angles, energies, temperature, lattice_size, 
               num_rows, parity, above_rank, below_rank, acceptance_ratio, step)

    # Recompute energies and update system properties
    block_energy(num_rows, angles, energies[1], offset%2)
    total_energy[step] = np.sum(energies[1])
    order_tensor[step] = partial_Q(angles, lattice_size)

    return angles, energies, total_energy, order_tensor, acceptance_ratio

def main(program_name, num_steps, lattice_size, temperature, plot_flag):
    """
    Arguments:
      program_name (string) = the name of the program;
      num_steps (int) = number of Monte Carlo steps (MCS) to perform;
      lattice_size (int) = side length of square lattice to simulate;
      temperature (float) = reduced temperature (range 0 to 2);
      plot_flag (int) = a flag to control plotting.
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    taskid = comm.Get_rank()
    numtasks = comm.Get_size()
    numworkers = numtasks-1
  
    # Create arrays to store system properties
    total_energy = np.zeros(num_steps)
    acceptance_ratio = np.zeros(num_steps)
    order_tensor = np.zeros((num_steps,2,2))

    if taskid == MASTER:
        # Check if numworkers is within range
        if (numworkers > MAXWORKER) or (numworkers < MINWORKER):
            print("ERROR: the number of tasks must be between %d and %d." % (MINWORKER+1,MAXWORKER+1))
            print("Quitting...")
            comm.Abort()

        # Initialize grid and energies
        angles = initdat(lattice_size)
        energies = np.zeros((lattice_size,lattice_size))

        # Plot initial state
        plotdat(angles, energies, plot_flag, lattice_size)

        # Distribute work to workers
        averow = lattice_size//numworkers
        extra = lattice_size%numworkers
        offset = 0

        initial_time = MPI.Wtime()
        for i in range(1, numworkers+1):
            rows = averow + (1 if i <= extra else 0)

            # Determine neighbor ranks
            above_rank = numworkers if i == 1 else i - 1
            below_rank = 1 if i == numworkers else i + 1

            # Send startup information
            comm.send(offset, dest=i, tag=BEGIN)
            comm.send(rows, dest=i, tag=BEGIN)
            comm.send(above_rank, dest=i, tag=BEGIN)
            comm.send(below_rank, dest=i, tag=BEGIN)
            comm.Send([angles[offset:offset+rows,:], rows*lattice_size, MPI.DOUBLE], dest=i, tag=BEGIN)
            offset += rows

        # Temporary arrays for collecting results
        temp_energy = np.zeros(num_steps)
        temp_order = np.zeros((num_steps,2,2))
        temp_acceptance = np.zeros(num_steps)
        
        # Collect results from workers
        for i in range(1, numworkers+1):
            offset = comm.recv(source=i, tag=DONE)
            rows = comm.recv(source=i, tag=DONE)
            comm.Recv([angles[offset,:], rows*lattice_size, MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([energies[offset,:], rows*lattice_size, MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([temp_energy[0:], num_steps, MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([temp_order[0:], num_steps*4, MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([temp_acceptance[0:], num_steps, MPI.DOUBLE], source=i, tag=DONE)
            
            total_energy += temp_energy
            order_tensor += temp_order
            acceptance_ratio += temp_acceptance

        # Calculate final order parameter and timing
        order = get_order(order_tensor, num_steps)
        final_time = MPI.Wtime()
        runtime = final_time-initial_time

        print(f"{program_name}: Size: {lattice_size}, Steps: {num_steps}, T*: {temperature:5.3f}: "
              f"Order: {order[num_steps-1]:5.3f}, Time: {runtime:8.6f} s")
        plotdat(angles, energies, plot_flag, lattice_size)

    elif taskid != MASTER:
        # Worker process code
        offset = comm.recv(source=MASTER, tag=BEGIN)
        num_rows = comm.recv(source=MASTER, tag=BEGIN)
        above_rank = comm.recv(source=MASTER, tag=BEGIN)
        below_rank = comm.recv(source=MASTER, tag=BEGIN)

        angles = np.zeros((num_rows+2, lattice_size))
        energies = np.zeros((2, num_rows, lattice_size))
        comm.Recv([angles[1,:], num_rows*lattice_size, MPI.DOUBLE], source=MASTER, tag=BEGIN)

        for step in range(num_steps):
            angles, energies, total_energy, order_tensor, acceptance_ratio = MC_step(
                comm, angles, energies, temperature, lattice_size, num_rows, 
                offset, above_rank, below_rank, total_energy, order_tensor, acceptance_ratio, step)
            
        # Send results back to master
        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(num_rows, dest=MASTER, tag=DONE)
        comm.Send([angles[1,:], num_rows*lattice_size, MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([energies[1,:,:], num_rows*lattice_size, MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([total_energy[0:], num_steps, MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([order_tensor[0:], num_steps*4, MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([acceptance_ratio[0:], num_steps, MPI.DOUBLE], dest=MASTER, tag=DONE)

if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        program_name = sys.argv[0]
        num_iterations = int(sys.argv[1])
        lattice_size = int(sys.argv[2])
        temperature = float(sys.argv[3])
        plot_flag = int(sys.argv[4])
        main(program_name, num_iterations, lattice_size, temperature, plot_flag)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))