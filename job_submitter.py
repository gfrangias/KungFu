import argparse
import subprocess

def main(args, job_name, output_name, error_name, num_nodes):
    # Create and configure the SLURM script
    slurm_script = f"""#!/bin/bash -l

#SBATCH --job-name={job_name}
#SBATCH --output={output_name}
#SBATCH --error={error_name}
#SBATCH --ntasks={num_nodes}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time={args.time}
#SBATCH --partition={args.partition}
#SBATCH --gres=gpu:{args.clients_distr}
#SBATCH --account={args.account}

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

num_clients="{args.clients}"
num_nodes="{num_nodes}"

nodelist=$(scontrol show hostnames $SLURM_NODELIST)

for node in $nodelist; do
    ip=$(host $node | awk '/has address/ {{ print $4 }}')
    if [ -z "$ip_list" ]; then
        ip_list="${{ip}}"
    else
        ip_list="${{ip_list}},${{ip}}"
    fi
done

echo "IP List: $ip_list"

srun python3 run_experiments.py --clients {args.clients} --nodes {num_nodes} --ips $ip_list --nic "eth0" --json {args.json}

"""
    # Save the SLURM script to a file
    with open(args.script_name, 'w') as f:
        f.write(slurm_script)

    # Submit the job using sbatch
    subprocess.run(['sbatch', args.script_name])

if __name__ == "__main__":

    # Get input parameters
    parser = argparse.ArgumentParser(description='Generate and submit a SLURM job')
    parser.add_argument('--special_name', default=None, help='Special naming for the job name')
    parser.add_argument('--clients', required=True, help='Number of clients')
    parser.add_argument('--clients_distr', required=True, help='Clients for each node.')
    parser.add_argument('--partition', default='gpu', help='SLURM partition where the tests will run')
    parser.add_argument('--account', default='pa230902', help='SLURM account')
    parser.add_argument('--time', default='01:00:00', help='Wall time of job')
    parser.add_argument('--json', required=True, help='name of JSON file that contains the experiments\' paramaters')
    parser.add_argument('--script_name', default='slurm_job.sh', help='Script name')

    args = parser.parse_args()

    # Generate SLURM job name
    if args.special_name is None:
        job_name = 'kf.cl'+args.clients
    else:
        job_name = 'kf.cl'+args.clients+'.'+args.special_name
    
    # Output files names
    output_name = job_name+'.out'
    error_name = job_name+'.err'

    # Calculate num_nodes
    num_nodes = int(args.clients) // int(args.clients_distr)

    # Check if there's a remainder
    remainder = int(args.clients) % int(args.clients_distr)

    # If there's a remainder, add 1 to the result
    if remainder != 0 or num_nodes == 0:
        num_nodes += 1

    print(f"{num_nodes} nodes will be used!")
    
    # Generate SLURM job script
    main(args, job_name, output_name, error_name, num_nodes)

