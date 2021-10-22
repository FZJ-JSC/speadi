from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from numpy import get_include as npinc

__version__ = '0.0.1'

setup(
    name='mylibrary',
    version=__version__,
    packages=find_packages(),
    author='Emile de Bruyn',
    author_email='e.de.bruyn@fz-juelich.de'
)
