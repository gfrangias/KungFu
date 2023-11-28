import argparse
import json

def generate_combinations(args):
    combinations = []
    default = {}
    default['exper_id'] = -1

    for _ in range(args.repetitions):

        for topologies in args.topologies:

            for clients in args.clients:
                
                for model in args.models:

                    for optimizer in args.optimizers:

                        for algorithm in args.algorithms:

                            # Dict default stores the default values of each parameter
                            default['clients'] = clients
                            default['topologies'] = topologies
                            default['algorithm'] = algorithm
                            default['model'] = model
                            default['optimizer'] = optimizer
                            default['threshold'] = args.thresholds[0]
                            if algorithm == "synchronous": default['threshold'] = 0.0
                            default['batch_size'] = args.batch_sizes[0]
                            default['epochs'] = args.epochs
                            default['exper_id']+=1

                            combinations.append(default.copy())

                            # Threshold is relevant only for FDA methods
                            if algorithm != "synchronous":
                                for threshold in args.thresholds[1:]:
                                    default['exper_id']+=1
                                    temp = default.copy()
                                    temp['threshold'] = threshold
                                    combinations.append(temp.copy())

                            for batch_size in args.batch_sizes[1:]:
                                default['exper_id']+=1
                                temp = default.copy()
                                temp['batch_size'] = batch_size
                                combinations.append(temp.copy())
        
    return combinations

if __name__ == "__main__":

    # Get the input arguments
    parser = argparse.ArgumentParser(description='Create a JSON file configuration of experiments parameters.')
    parser.add_argument('--name', type=str, default='no', help='(optional) give a custom name for the json file')
    parser.add_argument('--clients', type=int, nargs="+", help='number of clients in the network')
    parser.add_argument('--topologies', type=str, nargs="+", help='network topologies')
    parser.add_argument("--algorithms", type=str, nargs="+", help="algorithms used")
    parser.add_argument("--models", type=str, nargs="+", help='ANN models used')
    parser.add_argument("--epochs", type=int, help='number of epochs that the experiment will run')
    parser.add_argument("--thresholds", type=float, nargs="+", help='Thresholds to be used in FDA experiments. First add the default threshold!')
    parser.add_argument("--batch_sizes", type=int, nargs="+", help='Batch sizes to be used in experiments. First add the default batch size!')
    parser.add_argument("--optimizers", type=str, nargs="+", help='Optimizer used')
    parser.add_argument("--repetitions", type=int, default=1, help='How many times to repeat the same experiments.')
    parser.add_argument('--print', action="store_true", help="print json")
    args = parser.parse_args()
    
    # Generate all possible combinations
    combinations = generate_combinations(args)
    
    # Generate the JSON file name
    if args.name == 'no':
        json_file_name = "json_experiments/experiments_"+"_".join([str(clients) for clients in args.clients])+".json"
    else:
        json_file_name = "json_experiments/" + args.name

    # If --print just print the parameters combinations
    if args.print:
        print('\n'.join(map(str, combinations)))
    # Else write the combinations in the JSON file
    else:
        # Save to JSON file
        with open(json_file_name, 'w', encoding='utf-8') as f:
            json.dump(combinations, f, indent=4)
            print(str(len(combinations))+" experiments have been saved in file: "+json_file_name)

