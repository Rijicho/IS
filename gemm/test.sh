#!/bin/bash

source /opt/intel/bin/compilervars.sh intel64
source /var/mpi-selector/data/openmpi-1.10.4-intel-v16.0.3.sh

export OMP_NUM_THREADS=20

./main
