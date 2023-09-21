import argparse, itertools, subprocess
from run_tests_functions import map_epochs, get_sublist

parser = argparse.ArgumentParser(description='Run multiple KungFu commands')
parser.add_argument('--nodes', type=int, required=True,
                    help='number of nodes in the network')
parser.add_argument('--ips', type=str, required=True,
                    help='a list of ips in the form: IP1:slots,IP2:slots,...')
parser.add_argument('--nic', type=str, required=True,
                    help='the NIC of the network')
parser.add_argument('--index', type=int, default=None,
                    help='which task will perform these commands')
parser.add_argument('--print', action="store_true", help="print commands")
args = parser.parse_args()

exper_type_list = ["Synchronous SGD"]
model_type_list = ["lenet5"]
threshold_list = ["0.5", "1", "1.5", "2"]
batch_size_list = ["32", "64", "128", "256"]

commands = []

for exper_type in exper_type_list:

    if exper_type == "Naive FDA":
        # Use subprocess to run the command
        combinations = list(itertools.product(threshold_list, batch_size_list, model_type_list))
        for model_type, threshold, batch_size in combinations:

            command = "kungfu-run -np "+str(args.nodes)+ \
                " -H "+ args.ips +" -nic "+ args.nic + \
                " python3 examples/fda_examples/tf2_mnist_naive_fda.py --epochs "+map_epochs(exper_type, model_type, batch_size, threshold)+" -l "\
                "--model "+ model_type +" --threshold "+ threshold +" --batch "+batch_size 
            commands.append(command)

    elif exper_type == "Synchronous SGD":
        combinations = list(itertools.product(batch_size_list, model_type_list))
        for model_type, batch_size in combinations:

            command = "kungfu-run -np "+str(args.nodes)+ \
                " -H "+ args.ips +" -nic "+ args.nic + \
                " python3 examples/fda_examples/tf2_mnist_sync_sgd.py --epochs "+map_epochs(exper_type, model_type, batch_size)+" -l "\
                "--model "+ model_type +" --batch "+batch_size 

            commands.append(command)

total_commands = len(commands)
if args.index is not None:
    commands = get_sublist(commands, args.index)

print("Running "+ str(total_commands) +" experiments on "+ str(args.nodes) +" nodes!")

for i, command in enumerate(commands):
    print(f"On this task: command {i + 1}/{len(commands)}")
    print(command)
    if not args.print:
        subprocess.run(command, shell=True)
        subprocess.run("sleep 1m", shell=True)



