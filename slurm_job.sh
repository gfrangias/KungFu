#!/bin/bash -l

#SBATCH --job-name=kf.cl8
#SBATCH --output=kf.cl8.out
#SBATCH --error=kf.cl8.err
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --account=pa230902

module purge

module load gnu/8
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load tensorflow/2.4.1
module load cmake/3.7.2
conda activate kungfu-aris
export PATH=$HOME/.local/bin:$PATH

export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"

## RUN YOUR PROGRAM ##
ip_list=""

num_clients="8"
num_nodes="4"

nodelist=$(scontrol show hostnames $SLURM_NODELIST)

for node in $nodelist; do
    ip=$(host $node | awk '/has address/ { print $4 }')
    if [ -z "$ip_list" ]; then
        ip_list="${ip}"
    else
        ip_list="${ip_list},${ip}"
    fi
done

echo "IP List: $ip_list"

srun python3 run_experiments.py --clients $num_clients --nodes $num_nodes --ips $ip_list --nic "eth0" --json experiments_8.json

