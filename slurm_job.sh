#!/bin/bash -l

#SBATCH --job-name=kf.cl4
#SBATCH --output=kf.cl4.out
#SBATCH --error=kf.cl4.err
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --account=pa230902

module purge
module load gnu/8
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load python/3.8.13
module load tftorch/270-191

export PATH=$HOME/.local/bin:$PATH
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"

## RUN YOUR PROGRAM ##
ip_list=""

num_clients="4"
num_nodes="2"

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

srun python3 run_experiments.py --clients 4 --nodes 2 --ips $ip_list --nic "eth0" --json experiments_4.json

