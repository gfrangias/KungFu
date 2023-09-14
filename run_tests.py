import argparse, itertools

parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--nodes', type=int, default=4,
                    help='number of nodes in the network')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs')
args = parser.parse_args()

def ips_to_string(ips):
    last_ip = ips[-1]+":1"
    ips = [ip+":1," for ip in ips[:-1]]
    ips.append(last_ip)
    ips = ''.join(ips)
    return ips

exper_type_list = ["Synchronous SGD", "Naive FDA"]
model_type_list = ["lenet5", "adv_cnn"]
threshold_list = ["0.5", "1", "1.5", "2"]
batch_size_list = ["32", "64", "128"]

commands = []

### ???
ips = ["83.212.80.31","83.212.80.45","83.212.80.103","83.212.80.107"]
nic = "eth1"
### ???

ips = ips_to_string(ips)

for exper_type in exper_type_list:

    if exper_type == "Naive FDA":
        # Use subprocess to run the command
        combinations = list(itertools.product(model_type_list, threshold_list, batch_size_list))
        for model_type, threshold, batch_size in combinations:

            command = "kungfu-run -np "+str(args.nodes)+ \
                " -H "+ ips +" -nic "+ nic + \
                " python3 examples/fda_examples/tf2_mnist_naive_fda.py --epochs "+str(args.epochs)+" -l "\
                "--model "+ model_type +" --threshold "+ threshold +" --batch "+batch_size 
            commands.append(command)

    elif exper_type == "Synchronous SGD":
        combinations = list(itertools.product(model_type_list, batch_size_list))
        for model_type, batch_size in combinations:

            command = "kungfu-run -np "+str(args.nodes)+ \
                " -H "+ ips +" -nic "+ nic + \
                " python3 examples/fda_examples/tf2_mnist_sync_sgd.py --epochs "+str(args.epochs)+" -l "\
                "--model "+ model_type +" --batch "+batch_size 

            commands.append(command)

print("Running "+ str(len(commands)) +" experiments on "+ str(args.nodes) +" nodes!")
[print(command) for command in commands]
