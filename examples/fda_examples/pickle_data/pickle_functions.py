import pickle, os
import tensorflow as tf



def initialize_logs(exper_type, model_type, nodes, total_epochs, batch_size):
    training_logs = {"epoch": [], "step": [], "syncs": [], "batch_loss": [], "epoch_accuracy": []}
    training_logs["loss"] = 1
    training_logs["accuracy"] = 0
    training_logs["duration"] = 0
    training_logs["exper_type"] = exper_type
    training_logs["model_type"] = model_type
    training_logs["nodes"] = nodes
    training_logs["total_epochs"] = total_epochs
    training_logs["batch_size"] = batch_size
    
    return training_logs

def step_update_logs(training_logs, step, steps_per_epoch, syncs, batch_loss):
    training_logs["epoch"].append(int(step / steps_per_epoch + 1))
    training_logs["step"].append(step)
    training_logs["syncs"].append(syncs)
    training_logs["batch_loss"].append(batch_loss.numpy())

    return training_logs

def epoch_update_accuracy(training_logs, epoch_accuracy):
    training_logs["epoch_accuracy"].append(epoch_accuracy)

    return training_logs

def store_pickle(training_logs, loss, accuracy, duration):
    training_logs["loss"] = loss
    training_logs["accuracy"] = accuracy
    training_logs["duration"] = duration

    base_filename = training_logs["exper_type"].replace(" ", "_") + "." + training_logs["model_type"] + ".nodes-" +\
    str(training_logs["nodes"]) + ".epochs-" + str(training_logs["total_epochs"]) + ".batch-" + str(training_logs["batch_size"]) + ".pkl"
    
    filename = base_filename
    n = 1

    # Check if file exists, if yes, add (n) to its name
    while os.path.exists(os.path.dirname(os.path.abspath(__file__))+ "/" + filename):
        filename = base_filename.replace('.pkl', f'({n}).pkl')
        n += 1

    with open(os.path.dirname(os.path.abspath(__file__))+ "/" + filename, "wb") as f:
        pickle.dump(training_logs, f)

    print("Data saved at: " + os.path.dirname(os.path.abspath(__file__)) + "/" + filename)

def edit_pickle(filename, field, value):
    # Load the data from the pickle file
    data = load_pickle(filename)
    
    # Check if the field exists in the data, if not raise an exception
    if field not in data:
        raise KeyError(f"'{field}' does not exist in the loaded data.")
    
    # Update the field with the new value
    data[field] = value

    # Save the updated data back to the pickle file
    with open(os.path.dirname(os.path.abspath(__file__)) + "/" + filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Updated '{field}' in {filename} with value: {value}")

def load_pickle(filename):

    with open(os.path.dirname(os.path.abspath(__file__)) + "/" + filename, "rb") as f:
        loaded_logs = pickle.load(f)
    
    print(loaded_logs)
    
    return loaded_logs