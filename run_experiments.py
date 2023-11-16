import argparse, subprocess, json
from run_experiments_fns import distribute_clients

# Get the input arguments
parser = argparse.ArgumentParser(description='Run multiple KungFu commands')
parser.add_argument('--clients', type=int, required=True,
                    help='number of clients in the network')
parser.add_argument('--nodes', type=int, required=True,
                    help='number of nodes in the network')
parser.add_argument('--ips', type=str, required=True,
                    help='a list of ips in the form: IP1,IP2,...')
parser.add_argument('--nic', type=str, required=True,
                    help='the NIC of the network')
parser.add_argument('--json', type=str, required=True,
                    help='name of JSON file that contains the experiments\' paramaters')
parser.add_argument('--print', action="store_true", help="print commands")
args = parser.parse_args()

# Initialize empty list of commands
commands = []

# Get the input IPs in a list
ips_list = args.ips.split(',')

# Distribute the clients to physical nodes
clients_distr = distribute_clients(args.clients, args.nodes)
print("Clients at each node: " +str(clients_distr))

# Create the string equivalent of the IP list for KungFu command
ips = [f"{ips_list[i]}:{clients_distr[i]}" for i in range(args.nodes)]
ips = ",".join(ips)

# Open the JSON file for reading
with open('json_experiments/'+args.json, 'r') as f:
    experiments = json.load(f)

# For every experiment in the JSON file
for experiment in experiments:

    # Create the KungFu command
    command = "kungfu-run -np %d -H %s --nic %s python3 examples/fda_examples/tf2_mnist_experiment.py --epochs %d --model %s --batch %d --threshold %.2f --algorithm %s --clients_distr \"%s\" --optimizer %s -l" % \
        (args.clients, ips, args.nic, experiment['epochs'], experiment['model'], experiment['batch_size'], experiment['threshold'], experiment['algorithm'], str(clients_distr), experiment['optimizer'])
    
    # If --print then just print the commands
    if args.print:
        print(command)
    # Else run the command in a subprocess and then sleep for 1 minute
    else:
        subprocess.run(command, shell=True)
        subprocess.run("sleep 1m", shell=True)

