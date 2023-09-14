import pandas as pd

class data_analysis:

    def __init__(self):

        self.directory = "examples/fda_examples/parquet_files/"
        self.files = [self.directory + "info.parquet", self.directory + "step.parquet", \
                      self.directory + "epoch.parquet"]
        self.dfs = [pd.read_parquet(file) for file in self.files]

    

    def print_exper(self, exper_id):

        results = [df.query(f"exper_id == {exper_id}") for df in self.dfs]

        for result in results: print(result+"\n")

    def print_info(self):
        
        print(self.dfs[0])

    def print_all(self):
        pd.set_option('display.max_columns', None)
        for df in self.dfs: print(df)
        pd.reset_option('display.max_columns')
