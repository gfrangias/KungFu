import pandas as pd
import os

class logs_df:

    def __init__(self, logs_dict):

        self.directory = "examples/fda_examples/parquet_files/"
        self.info_file = self.directory + "info.parquet"
        self.step_file = self.directory + "step.parquet"
        self.epoch_file = self.directory + "epoch.parquet"
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

        if os.path.exists(self.info_file) and os.path.exists(self.step_file) and os.path.exists(self.epoch_file):
            info_df = pd.read_parquet(self.info_file)
            step_df = pd.read_parquet(self.step_file)
            epoch_df = pd.read_parquet(self.epoch_file)

            combined_info = pd.concat([info_df, self.info_df])
            combined_step = pd.concat([step_df, self.step_df])
            combined_epoch = pd.concat([epoch_df, self.epoch_df])

        else:
            combined_info = self.info_df
            combined_step = self.step_df
            combined_epoch = self.epoch_df    
            
        combined_info.to_parquet(self.info_file)
        combined_step.to_parquet(self.step_file)
        combined_epoch.to_parquet(self.epoch_file)
        print("Stored with experiment ID: "+str(self.id))
        print("In files:\t"+"info.parquet")
        print("\t\t"+"step.parquet")
        print("\t\t"+"epoch.parquet")

    def load_parquet(self):
        dfs = pd.read_parquet(parquet_file, engine='pyarrow', storage_options=None)
        print(dfs)