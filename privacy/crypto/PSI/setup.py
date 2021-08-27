from distutils.core import setup, Extension
import numpy
import os, re
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

example_module = Extension('_PSI',
                           sources=['PSI_wrap.cxx','src/PsiReceiver.cpp', 'src/PsiSender.cpp', 'src/utils.cpp',
                                    'src/AES.cpp', 'src/PRNG.cpp', 'src/RandomOracle.cpp',
                                    # 'blake2/blake2b.cpp', 'blake2/blake2bp.cpp', 'blake2/blake2xb.cpp'
                                    ],
                           include_dirs=["./blake2/",
                                         "./include",
                                         numpy_include],

                           extra_compile_args=['-std=c++17', '-march=native','-maes', '-O3', '-Wall',  '-msse2', '-msse', ],
                           # extra_objects = ["./blake2/libblake.a"]
                           )
for file in os.listdir("./"):
    if file.find("_PSI") != -1:
        os.remove(file)
        break

setup (
    name = 'PSI',
    version = '0.1',
    ext_modules = [example_module],
    py_modules = ["PSI"],
)

for file in os.listdir("./"):
    if file.find("_PSI") != -1:
        os.rename(file, "_PSI.so")
        break
