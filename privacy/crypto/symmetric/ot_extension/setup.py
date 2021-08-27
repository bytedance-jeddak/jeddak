from distutils.core import setup, Extension
import numpy
import os
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

example_module = Extension('_iknpOTe',
                           sources=['iknpOTe_wrap.cxx','src/iknpOTeReceiver.cpp', 'src/iknpOTeSender.cpp', 'src/utils.cpp',
                                    'src/AES.cpp', 'src/PRNG.cpp', 'src/RandomOracle.cpp',
                                    ],
                           include_dirs=["./blake2/",
                                         "./include",
                                         numpy_include],
                           extra_compile_args=['-std=c++17', '-march=native','-maes', '-O3', '-Wall',  '-msse2'],
                           )
for file in os.listdir("./"):
    if file.find("_iknpOTe") != -1:
        os.remove(file)
        break

setup (
    name = 'iknpOTe',
    version = '0.1',
    ext_modules = [example_module],
    py_modules = ["iknpOTe"],
)

for file in os.listdir("./"):
    if file.find("_iknpOTe") != -1:
        os.rename(file, "_iknpOTe.so")
        break
