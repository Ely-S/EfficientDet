from distutils.core import setup

from Cython.Build import cythonize
import numpy

setup(name='EfficientDet',
      ext_modules=cythonize("utils/compute_overlap.pyx"),
      include_dirs=[numpy.get_include()]
)