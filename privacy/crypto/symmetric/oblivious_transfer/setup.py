from distutils.core import setup, Extension
import os

source_files = []
include_files = []
for file in os.listdir("./source"):
    if file.split(".")[-1] == 'c':
        source_files.append(os.path.join("./source", file))

example_module = Extension('libecc',
                           sources=source_files,
                           include_dirs=["./include"],
                           extra_compile_args=['-O3'],
                           )
for file in os.listdir("./"):
    if file.find("libecc") != -1:
        os.remove(file)
        break

setup (
    name = 'libecc',
    version = '0.1',
    ext_modules = [example_module],
    py_modules = ["libecc"],
)

for file in os.listdir("./"):
    if file.find("libecc") != -1:
        os.rename(file, "libecc.so")
        break

