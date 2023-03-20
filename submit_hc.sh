#!/bin/bash
# Specify a partition 
#SBATCH --partition=dggpu
# Request nodes 
#SBATCH --nodes=1
# Request some processor cores 
#SBATCH --ntasks=4
# Request GPUs 
#SBATCH --gres=gpu:1
# Request memory 
#SBATCH --mem=25G
# Maximum runtime of 2 hours
#SBATCH --time=20:00:00

python begin_hc.py $1 $2 $3