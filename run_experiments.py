import argparse, subprocess, json
from run_experiments_fns import distribute_clients

parser = argparse.ArgumentParser(description='Run multiple KungFu commands')
parser.add_argument('--clients', type=int, required=True,
                    help='number of clients in the network')
parser.add_argument('--nodes', type=int, required=True,
                    help='number of nodes in the network')
parser.add_argument('--ips', type=str, required=True,
                    help='a list of ips in the form: IP1:slots,IP2:slots,...')
parser.add_argument('--nic', type=str, required=True,
                    help='the NIC of the network')
parser.add_argument('--json', type=str, required=True,
                    help='name of JSON file that contains the experiments\' paramaters')
parser.add_argument('--print', action="store_true", help="print commands")
args = parser.parse_args()

commands = []

ips_list = args.ips.split(',')
clients_per_node = distribute_clients(args.clients, args.nodes)
print("Clients at each node: " +str(clients_per_node))

ips = [f"{ips_list[i]}:{clients_per_node[i]}" for i in range(args.nodes)]
ips = ",".join(ips)

# Open the file for reading
with open('json_experiments/'+args.json, 'r') as f:
    experiments = json.load(f)

for experiment in experiments:

    command = "kungfu-run -np %d -H %s python3 examples/fda_examples/tf2_mnist_experiment --epochs %d --model %s --batch %d --threshold %.2f --exper_type %s -l" % \
        (args.clients, ips, experiment['epochs'], experiment['model'], experiment['batch_size'], experiment['threshold'], experiment['algorithm'])
    
    if args.print:
        print(command)
    else:
        subprocess.run(command, shell=True)
        subprocess.run("sleep 1m", shell=True)