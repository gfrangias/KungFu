from datetime import datetime
import os
import pandas as pd
import tensorflow as tf

class logs_dict:

    def __init__(self, algorithm, model, clients, clients_distr, threshold, batch_size, epoch_steps, epochs):

        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

        self.info_data = {
            "exper_id" : None,
            "algorithm" : algorithm,
            "model" : model,
            "clients" : clients,
            "clients_distr" : clients_distr,
            "epochs" : epochs,
            "threshold" : threshold,
            "batch_size" : batch_size,
            "syncs" : 0,
            "duration" : 0,
            "timestamp" : timestamp,
            "epoch_steps" : epoch_steps
        }

        self.epoch_data = {
            "epoch" : [],
            "steps" : [],
            "syncs" : [],
            "accuracy" : [],
            "loss" : [],
            "time" : [],
            "com_time" : [],
            "calc_time" : []
        }

        self.step_data = {

            "step" : [],
            "epoch" : [],
            "syncs" : [],
            "loss" : [],
            "time" : [],
            "com_time" : [],
            "calc_time" : []
        }
    
    def step_update(self, step, epoch, syncs, loss, time, com_time, calc_time):
        self.step_data["step"].append(step)
        self.step_data["epoch"].append(int(epoch))

        if  tf.is_tensor(syncs):
            self.step_data["syncs"].append(syncs.numpy())
        else:
            self.step_data["syncs"].append(syncs)
        
        self.step_data["loss"].append(loss.numpy())
        self.step_data["time"].append(time)
        self.step_data["com_time"].append(com_time)
        self.step_data["calc_time"].append(calc_time)

    def epoch_update(self, epoch, steps, syncs, accuracy, loss, time, com_time, calc_time):
        self.epoch_data["steps"].append(steps)
        self.epoch_data["epoch"].append(epoch)

        if  tf.is_tensor(syncs):
            syncs = syncs.numpy()

        self.epoch_data["syncs"].append(syncs)
        self.info_data["syncs"] = syncs
        self.epoch_data["accuracy"].append(accuracy)
        self.epoch_data["loss"].append(loss)
        self.epoch_data["time"].append(time)
        self.epoch_data["com_time"].append(com_time)
        self.epoch_data["calc_time"].append(calc_time)
        self.info_data["duration"] = time


    def id_update(self):
        if os.path.exists("examples/fda_examples/csv_files/latest_expers/info.csv"):
            df = pd.read_csv("examples/fda_examples/csv_files/latest_expers/info.csv")
            exper_id = df["exper_id"].max() + 1
        else:
            exper_id = 0

        self.info_data["exper_id"] = exper_id
