import pandas as pd

class data_analysis:

    def __init__(self):

        self.directory = "examples/fda_examples/parquet_files/"
        self.files = [self.directory + "info.parquet", self.directory + "step.parquet", \
                      self.directory + "epoch.parquet"]
        self.dfs = [pd.read_parquet(file) for file in self.files]

    def print_exp(self, exper_id):

        results = [df.query(f"exper_id == {exper_id}") for df in self.dfs]

        for result in results: print(result+"\n")