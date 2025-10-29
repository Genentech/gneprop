#!/bin/bash

# Set up conda
conda config --set solver libmamba

# Create and activate conda environment
conda env create -f environment.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gneprop

# Build learn2learn
git clone https://github.com/learnables/learn2learn.git
cd learn2learn
make build
python setup.py bdist_wheel
cd ..

# Build GNEpropCPP
conda install -y pybind11 libboost-devel
cd GNEpropCPP
python setup.py bdist_wheel
cd ..

# Install learn2learn and GNEpropCPP
L2L_WHL=$(ls learn2learn/dist/learn2learn-*-cp311-cp311-linux_x86_64.whl)
GNECPP_WHL=$(ls GNEpropCPP/dist/gnepropcpp-*-cp311-cp311-linux_x86_64.whl)

pip list | awk '{print$1"=="$2}' | tail -n +3 > base_constraints.txt
pip install -c base_constraints.txt $L2L_WHL
pip install -c base_constraints.txt $GNECPP_WHL
