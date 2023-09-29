import argparse
import json

def generate_combinations(args):
    combinations = []
    default = {}
    default['exper_id'] = -1

    for clients in args.clients:
        
        for algorithm in args.algorithms:

            for model in args.models:
                default['clients'] = clients
                default['algorithm'] = algorithm
                default['model'] = model
                default['threshold'] = args.thresholds[0]
                if algorithm == "synchronous": default['threshold'] = 0.0
                default['batch_size'] = args.batch_sizes[0]
                default['epochs'] = args.epochs
                default['exper_id']+=1

                combinations.append(default.copy())

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
    parser = argparse.ArgumentParser(description='Create a JSON file configuration of experiments parameters.')
    parser.add_argument('--clients', type=int, nargs="+", help='number of clients in the network')
    parser.add_argument("--algorithms", type=str, nargs="+", help="algorithms used")
    parser.add_argument("--models", type=str, nargs="+", help='ANN models used')
    parser.add_argument("--epochs", type=int, help='number of epochs that the experiment will run')
    parser.add_argument("--thresholds", type=float, nargs="+", help='Thresholds to be used in FDA experiments. First add the default threshold!')
    parser.add_argument("--batch_sizes", type=int, nargs="+", help='Batch sizes to be used in experiments. First add the default batch size!')
    parser.add_argument('--print', action="store_true", help="print json")
    args = parser.parse_args()
    
    combinations = generate_combinations(args)
    
    json_file_name = "json_experiments/experiments_"+"_".join([str(clients) for clients in args.clients])+".json"

    # Save to JSON file
    with open(json_file_name, 'w') as f:
        json.dump(combinations, f, indent=4)

    print(str(len(combinations))+" experiments have been saved in file: "+json_file_name)
