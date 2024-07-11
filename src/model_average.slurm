#!/bin/bash

#SBATCH --job-name=<name>
#SBATCH --mail-type=<status>
#SBATCH --mail-user=<email>

#SBATCH --account=<lab>
#SBATCH --partition=<node_type>
#SBATCH --nodes=<1>
#SBATCH --ntasks-per-node=<6>
#SBATCH --mem=<150g>
#SBATCH --gpus=<type:quantity> 
#SBATCH --time=<time> # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=<working directory>
#SBATCH --export=all
#SBATCH --output=<file> # where STDOUT goes
#SBATCH --error=<file> # where STDERR goes

# Modules to use (optional).
<e.g., module load apptainer>

# Your programs to run.
<my_programs>