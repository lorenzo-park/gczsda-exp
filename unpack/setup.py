from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="unpackfloat",
    ext_modules=cythonize("unpackfloat.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
