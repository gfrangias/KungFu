#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=initial_test    # Job name
#SBATCH --output=initial_test.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=initial_test.%j.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=1     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=20     # Threads per task
#SBATCH --mem=16G
#SBATCH --time=01:00:00   # walltime
#SBATCH --partition=compute    # Partition
#SBATCH --account=pa230401    # Replace with your system project


## LOAD MODULES ##
module purge		# clean up loaded modules 

# load necessary modules
module load gnu/8 
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load tensorflow/2.4.1 
module load cmake/3.7.2
export PATH=$HOME/.local/bin:$PATH

export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"

## RUN YOUR PROGRAM ##
srun kungfu-run -np 4 python3 examples/fda_examples/tf2_mnist_sync_sgd.py --epochs 100 -l --model adv_cnn --batch 64 
