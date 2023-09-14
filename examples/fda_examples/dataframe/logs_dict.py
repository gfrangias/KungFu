from datetime import datetime
import os
import pandas as pd
import tensorflow as tf

class logs_dict:

    def __init__(self, exper_type, model_type, nodes, threshold, batch_size, steps_per_epoch):
        
        if os.path.exists("examples/fda_examples/parquet_files/info.parquet"):
            df = pd.read_parquet("examples/fda_examples/parquet_files/info.parquet")
            exper_id = df["exper_id"].max() + 1
        else:
            exper_id = 0

        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

        self.info_data = {
            "exper_id" : exper_id,
            "exper_type" : exper_type,
            "model_type" : model_type,
            "nodes" : nodes,
            "threshold" : threshold,
            "batch_size" : batch_size,
            "timestamp" : timestamp,
            "steps_per_epoch" : steps_per_epoch
        }

        self.epoch_data = {
            "accuracy" : []
        }

        self.step_data = {

            "step" : [],
            "syncs" : [],
            "loss" : [],
        }
    
    def step_update(self, step, syncs, loss):
        self.step_data["step"].append(step)
        if  tf.is_tensor(syncs):
            self.step_data["syncs"].append(syncs.numpy())
        else:
            self.step_data["syncs"].append(syncs)
        self.step_data["loss"].append(loss.numpy())
    
    def epoch_update(self, accuracy):
        self.epoch_data["accuracy"].append(accuracy)

def find_next_id(directory):
    # Get a list of filenames in the directory
    filenames = os.listdir(directory)

    # Initialize a list to store the extracted IDs
    ids = []

    # Iterate through the filenames and extract the IDs
    for filename in filenames:
        # Check if the filename matches the format "exp_{id}_*.parquet"
        if filename.startswith("exp_"):
            try:
                # Extract the ID by splitting the filename and parsing it as an integer
                file_id = int(filename.split("_")[1])
                ids.append(file_id)
            except ValueError:
                pass  # Ignore filenames that don't have a valid ID

    # Find the maximum ID, or return 0 if no IDs were found
    if ids:
        max_id = max(ids)
    else:
        max_id = 0

    # Return the next integer ID
    return max_id + 1