from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Set OpenMP flags based on the operating system
if os.name == 'nt':  # Windows
    extra_compile_args = ['/openmp']
    extra_link_args = []
else:  # Unix-like systems
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        "LebwohlLasher_CMULT",
        ["LebwohlLasher_CMULT.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    )
)