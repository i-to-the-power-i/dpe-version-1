from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Resolve sources relative to this setup.py so the build works from any cwd.
extensions = [
    Extension(
        "model.cy_utils", 
        [os.path.join(PROJECT_ROOT, "src", "model", "cy_utils.pyx")], 
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "model.utils", 
        [os.path.join(PROJECT_ROOT, "src", "model", "utils.py")], 
        include_dirs=[numpy.get_include()]
    ),
    # Extension(
    #     "model.sequence_extractor", 
    #     ["src/model/sequence_extractor.py"], 
    #     include_dirs=[numpy.get_include()]
    # ),
    # Extension(
    #     "model.cy_sequence_extractor", 
    #     ["src/model/cy_sequence_extractor.pyx"], 
    #     include_dirs=[numpy.get_include()]
    # ),
]

setup(
    # force=True ensures everything is re-cythonized every time
    ext_modules=cythonize(
        extensions, 
        language_level="3", 
        annotate=True, 
        force=True
    )
)
