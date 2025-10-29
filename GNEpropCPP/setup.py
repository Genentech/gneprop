#!/usr/bin/python

"""
Setup script that builds GNEpropCPP
"""

from distutils.core import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch, rdkit, os


torch_dir = torch.__path__[0]
rdkit_lib_index = rdkit.__path__[0].split('/').index('lib')
rdkit_prefix = '/'.join(rdkit.__path__[0].split('/')[:rdkit_lib_index])

ext_modules = [
        Pybind11Extension('GNEpropCPP', sources=["GNEpropCPP.cpp"],
            language="c++",
            cxx_std=17,
            include_dirs = [os.path.join(torch_dir,"include"),
                os.path.join(torch_dir,"include/torch/csrc/api/include"),
                os.path.join(rdkit_prefix, "include/rdkit")],
            libraries = ["RDKitGraphMol", "RDKitSmilesParse", "torch_cpu", "torch_python"],
            library_dirs = [os.path.join(rdkit_prefix,"lib"),
                os.path.join(torch_dir,"lib")],
            extra_compile_args=["-O3"]
        )
]

setup(name = "GNEpropCPP",
    version = "0.1",
    author = "Nia Dickson",
    author_email="ndickson@nvidia.com",
    license="MIT",
    description = "C++ extension for GNEprop",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext})
