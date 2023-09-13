import pandas as pd
import numpy as np

class logs_df:

    def __init__(self, logs_dict):

        data_sync_map = {"lenet5" : 2, "adv_cnn" : 3}
        data_non_sync_map = {"lenet5" : 1, "adv_cnn" : 1}

        self.info_df = pd.DataFrame(logs_dict.info_data, index=[0])
        self.info_df["data_sync"] = self.info_df["model_type"].map(data_sync_map)
        self.info_df["data_non_sync"] = self.info_df["model_type"].map(data_non_sync_map)
        self.id = self.info_df["exper_id"].iloc[0]

        self.step_df = pd.DataFrame(logs_dict.step_data)
        #self.step_df["epoch"] = self.step_df["step"] / self.info_df["steps_per_epoch"].iloc[0]
        #self.step_df["data_transmission"] = self.step_df["syncs"] * self.info_df["data_sync"].iloc[0] + \
        #                            (self.step_df["step"] - self.step_df["syncs"]) * self.info_df["data_non_sync"].iloc[0]
        self.step_df["exper_id"] = self.info_df["exper_id"].iloc[0]

        self.epoch_df = pd.DataFrame(logs_dict.epoch_data)
        self.epoch_df["exper_id"] = self.info_df["exper_id"].iloc[0]

    def append_in_parquet(self):
        print(self.info_df)
        print(self.epoch_df)
        print(self.step_df)
        self.info_df.to_parquet("examples/fda_examples/parquet_files/exp_"+str(self.id)+"_info.parquet")
        self.step_df.to_parquet("examples/fda_examples/parquet_files/exp_"+str(self.id)+"_step.parquet")
        self.epoch_df.to_parquet("examples/fda_examples/parquet_files/exp_"+str(self.id)+"_epoch.parquet")
        print("Stored with experiment ID: "+str(self.id))
        print("In files:\t"+"exp_"+str(self.id)+"_info.parquet")
        print("\t\t"+"exp_"+str(self.id)+"_step.parquet")
        print("\t\t"+"exp_"+str(self.id)+"_epoch.parquet")

    def load_parquet(self):
        dfs = pd.read_parquet(parquet_file, engine='pyarrow', storage_options=None)
        print(dfs)