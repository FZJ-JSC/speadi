from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from numpy import get_include as npinc

ext_modules = [
    Extension(
        name = 'rt_mic_p',
        sources = ['vanhove/src/rt_mic_p.pyx',],
        extra_compile_args = ['-O2', '-march=native', '-ffast-math', '-fopenmp', f'-I{npinc()}'],
        extra_link_args = ['-fopenmp', '-lm']
    )
]

setup(
    name='vanhove',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    author='Emile de Bruyn',
    author_email='e.de.bruyn@fz-juelich.de'
)
