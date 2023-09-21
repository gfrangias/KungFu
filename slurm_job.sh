#!/bin/bash -l

##########################################################
#                   ARIS KungFu job                      #
#                                                        #
# Submit script: sbatch slurm_job.sh {number of nodes}   #
#                                                        #
##########################################################

#SBATCH --job-name=kf${0}    # Job name
#SBATCH --output=./logs/kf${0}.out # Stdout 
#SBATCH --error=.logs/kf${0}.err # Stderr 
#SBATCH --array=0-3     # Array of task IDs
#SBATCH --ntasks=20     # Number of tasks(processes)
#SBATCH --nodes=${0}     # Number of nodes requested
#SBATCH --ntasks-per-node=4     # Tasks per node
#SBATCH --cpus-per-task=5     # Threads per task
#SBATCH --mem=12G
#SBATCH --time=01:00:00   # walltime
#SBATCH --partition=compute    # Partition
#SBATCH --account=pa230401    # Replace with your system project


## LOAD MODULES ##
module purge            # clean up loaded modules

# load necessary modules
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

# Get the list of hostnames
nodelist=$(scontrol show hostnames $SLURM_NODELIST)

# Loop over each node
for node in $nodelist; do

    ip=$(host $node | awk '/has address/ { print $4 }')

    # Append to the list, with ":1" added after each IP
    if [ -z "$ip_list" ]; then
        ip_list="${ip}:1"
    else
        ip_list="${ip_list},${ip}:1"
    fi
done

# Print or use the IP list
echo "IP List: $ip_list"

# Run your program
for node in $(seq 1 ${0}); do
  srun --exclusive -N1 -n1 -o kf${0}_${node}_0.out -e kf${0}_${node}_0.err python3 run_tests.py --nodes ${0} --ips $ip_list --nic your_nic_value --index 0 &
  srun --exclusive -N1 -n1 -o kf${0}_${node}_1.out -e kf${0}_${node}_1.err python3 run_tests.py --nodes ${0} --ips $ip_list --nic your_nic_value --index 1 &
  srun --exclusive -N1 -n1 -o kf${0}_${node}_2.out -e kf${0}_${node}_2.err python3 run_tests.py --nodes ${0} --ips $ip_list --nic your_nic_value --index 2 &
  srun --exclusive -N1 -n1 -o kf${0}_${node}_3.out -e kf${0}_${node}_3.err python3 run_tests.py --nodes ${0} --ips $ip_list --nic your_nic_value --index 3 &
done
