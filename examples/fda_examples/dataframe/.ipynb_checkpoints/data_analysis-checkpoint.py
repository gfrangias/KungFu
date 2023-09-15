import pandas as pd

class data_analysis:

    def __init__(self):

        self.directory = "examples/fda_examples/csv_files/"
        self.files = [self.directory + "info.csv", self.directory + "step.csv", \
                      self.directory + "epoch.csv"]
        self.dfs = [pd.read_csv(file) for file in self.files]

    def print_exp(self, exper_id):

        results = [df.query(f"exper_id == {exper_id}") for df in self.dfs]

        for result in results: print(result+"\n")