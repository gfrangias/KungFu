import pandas as pd

class data_analysis:

    def __init__(self):

        self.directory = "examples/fda_examples/csv_files/"
        self.files = [self.directory + "info.csv", self.directory + "step.csv", \
                      self.directory + "epoch.csv"]
        self.dfs = [pd.read_csv(file) for file in self.files]
        self.num_of_exper = self.dfs[0].shape[0]

    def print_info(self):
        
        print(self.dfs[0])

    # Print the dataframe or part of the dataframe
    # low: the lower limit of exper_id to be printed  
    # high: the higher limit of exper_id to be printed
    # query: a string containing a query to be ran
    def print_df(self, low=0, high=None, query=None):
        pd.set_option('display.max_columns', None)

        if high is None: high = self.num_of_exper

        if query is None:

            results = [df.query(f"exper_id.between({low}, {high})") for df in self.dfs]        

        else:

            range = [df.query(f"exper_id.between({low}, {high})") for df in self.dfs]
            
            exper_ids = self.dfs[0].query(query)['exper_id']

            results = [df[df['exper_id'].isin(exper_ids)] for df in range]
        
        for result in results: print(result)
        
        pd.reset_option('display.max_columns')

        return results

    #def print_syncs_accuracy(self):

    def group_repeated_expers(self, attributes, label=None, print=False):

        # Group by multiple columns
        grouped = self.dfs[0].groupby(attributes)

        # Initialize an empty dictionary to store the result
        result = {}

        # Iterate through each group and collect the IDs
        for name, group in grouped:
            ids = group['exper_id'].tolist()
            
            # If a label is specified, use it to map the output
            if label:
                label_value = group[label].iloc[0]
                result[label_value] = ids
            else:
                result[name] = ids

        if print: print(result)
        return(result)