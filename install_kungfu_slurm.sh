#!/bin/bash -l

#SBATCH --job-name=kf.install
#SBATCH --output=kf.install.out
#SBATCH --error=kf.install.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --account=pa230902

module purge

module load gnu/8
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load python/3.8.13
module load tftorch/270-191

conda activate kungfu-aris

pip3 install --no-index -U --user .
