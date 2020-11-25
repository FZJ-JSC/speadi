from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from numpy import get_include as npinc

ext_modules = [
    Extension("rt_mic_p",
              ["rt_mic_p.pyx"],
              extra_compile_args = ["-O3", "-fopenmp", f"-I{npinc()}", "-lm"],
              extra_link_args = ["-fopenmp", "-lm"]
             )
]
setup(name="Raw distance differences by time",
      ext_modules=cythonize(ext_modules),
     )
