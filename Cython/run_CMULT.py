"""
Run file for Lebwohl-Lasher simulation using Cython-optimized code with multi-threading
"""

import sys
from LebwohlLasher_CMULT import main

if __name__ == "__main__":
    if len(sys.argv) == 6:
        try:
            # Get command line arguments
            iterations = int(sys.argv[1])
            size = int(sys.argv[2])
            temperature = float(sys.argv[3])
            plotflag = int(sys.argv[4])
            threads = int(sys.argv[5])
            
            # Input validation
            if size <= 0 or iterations <= 0:
                raise ValueError("Size and iterations must be positive")
            if temperature < 0:
                raise ValueError("Temperature must be non-negative")
            if plotflag not in [0, 1, 2]:
                raise ValueError("Plot flag must be 0, 1, or 2")
            if threads <= 0:
                raise ValueError("Number of threads must be positive")
            
            # Run simulation
            main("LebwohlLasher", iterations, size, temperature, plotflag, threads)
            
        except ValueError as e:
            print(f"Error: {e}")
            print("\nUsage: python run_CMULT.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>")
            print("  ITERATIONS: positive integer (number of Monte Carlo steps)")
            print("  SIZE: positive integer (lattice size)")
            print("  TEMPERATURE: positive float (reduced temperature)")
            print("  PLOTFLAG: 0 (no plot), 1 (energy plot), or 2 (angle plot)")
            print("  THREADS: positive integer (number of threads to use)")
            sys.exit(1)
    else:
        print("\nUsage: python run_CMULT.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>")
        print("  ITERATIONS: positive integer (number of Monte Carlo steps)")
        print("  SIZE: positive integer (lattice size)")
        print("  TEMPERATURE: positive float (reduced temperature)")
        print("  PLOTFLAG: 0 (no plot), 1 (energy plot), or 2 (angle plot)")
        print("  THREADS: positive integer (number of threads to use)")
        sys.exit(1)