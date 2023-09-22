#!/bin/bash -l

##########################################################
#                   ARIS KungFu job                      #
#                                                        #
# Submit script: sbatch slurm_job.sh {number of nodes}   #
#                                                        #
##########################################################

# Compute the number of nodes needed

num_clients="$1"

num_nodes=$((num_clients / 4))

# Check if there's a remainder
remainder=$((num_clients % 4))

# If there's a remainder, add 1 to the result
if [ $remainder -ne 0 ]; then
        num_nodes=$((num_nodes + 1))
fi

echo "$num_nodes nodes will be used!"


#SBATCH --job-name=kf${num_clients}     # Job name
#SBATCH --output=kf${num_clients}.out   # Stdout (%j expands to jobId)
#SBATCH --error=kf${num_clients}.err    # Stderr (%j expands to jobId)
#SBATCH --ntasks=${num_nodes}           # Number of tasks(processes)
#SBATCH --nodes=${num_nodes}            # Number of nodes requested
#SBATCH --ntasks-per-node=1             # Tasks per node
#SBATCH --cpus-per-task=20              # Threads per task
#SBATCH --mem=32G
#SBATCH --time=01:00:00                 # walltime
#SBATCH --partition=compute             # Partition
#SBATCH --account=pa230902              # Replace with your system project

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
        ip_list="${ip}"
    else
        ip_list="${ip_list},${ip}"
    fi
done

# Print or use the IP list
echo "IP List: $ip_list"

srun run_tests --clients $num_clients --nodes $num_nodes --ips $ip_list --nic "eth0"