import pandas as pd
import os

class logs_df:

    def __init__(self, logs_dict):

        self.directory = "examples/fda_examples/csv_files/latest_expers/"
        self.info_file = self.directory + "info.csv"
        self.step_file = self.directory + "epoch_step_info/" + str(logs_dict.info_data["exper_id"]) + "_step.csv"
        self.epoch_file = self.directory + "epoch_step_info/" + str(logs_dict.info_data["exper_id"]) + "_epoch.csv"

        self.info_df = pd.DataFrame(logs_dict.info_data, index=[0])
        self.id = self.info_df["exper_id"].iloc[0]

        self.step_df = pd.DataFrame(logs_dict.step_data)
        self.step_df["exper_id"] = self.info_df["exper_id"].iloc[0]

        self.epoch_df = pd.DataFrame(logs_dict.epoch_data)
        self.epoch_df["exper_id"] = self.info_df["exper_id"].iloc[0]

    def append_in_csv(self):
        print(self.info_df)
        print(self.epoch_df)
        print(self.step_df)

        if os.path.exists(self.info_file):
            info_df = pd.read_csv(self.info_file)
            print("info.csv")
            print(info_df)
            combined_info = pd.concat([info_df, self.info_df])
        else:
            combined_info = self.info_df
  
        print(combined_info)
        combined_info.to_csv(self.info_file, index=False)
        self.step_df.to_csv(self.step_file, index=False)
        self.epoch_df.to_csv(self.epoch_file, index=False)
        print("Stored with experiment ID: "+str(self.id))
        print("In files:\t"+"info.csv")
        print("\t\t"+"step.csv")
        print("\t\t"+"epoch.csv")