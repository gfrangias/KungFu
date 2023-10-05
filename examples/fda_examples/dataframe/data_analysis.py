import pandas as pd

class data_analysis:

    def __init__(self):

        self.directory = "../csv_files/"
        self.files = [self.directory + "info.csv", self.directory + "step.csv", \
                      self.directory + "epoch.csv"]
        self.dfs = [pd.read_csv(file) for file in self.files]
        self.num_of_exper = self.dfs[0].shape[0]
        self.epoch_step_with_info()

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

    def group_repeated_expers(self, attributes, table=None):

        grouped_list = []

        if table is None:
            dfs = self.dfs
        else:
            dfs = [self.dfs[table]]

        # Group by multiple columns
        for df in dfs:
            grouped_list.append(df.groupby(attributes))

        self.grouped_list = grouped_list
    
    def get_values_from_id(self, table, attributes, id):

        result = self.dfs[table].loc[self.dfs[table]['exper_id'] == id, attributes]

        return result
    
    def min_max_mean_of_rows(self, table, attributes, ids):

        result_dict = {}

        for labels, id_list in ids.items():
            temp_dfs = []
            for id in id_list:
                result = self.get_values_from_id(table, attributes, id)
                temp_dfs.append(result.set_index(attributes[0]))
            
            # Combine the DataFrames
            combined_df = pd.concat(temp_dfs, keys=range(len(temp_dfs)))

            # Calculate the mean accuracy for each epoch
            agg_df = combined_df.groupby(attributes[0]).agg(
                mean=(attributes[1], 'mean'),
                min=(attributes[1], 'min'),
                max=(attributes[1], 'max')
                ).reset_index()

        return result_dict, attributes
    
    def get_last_epoch(self):
        # Find the maximum value in the 'epoch' column
        max_epoch = self.dfs[2]['epoch'].max()

        # Filter the DataFrame to get rows where 'epoch' is equal to the maximum value
        self.dfs[1] = self.dfs[1][self.dfs[1]['epoch'] == max_epoch]
        self.dfs[2] = self.dfs[2][self.dfs[2]['epoch'] == max_epoch]

    def epoch_step_with_info(self):
        
        self.dfs[1] = pd.merge(self.dfs[0], self.dfs[1], on='exper_id')
        self.dfs[2] = pd.merge(self.dfs[0], self.dfs[2], on='exper_id')


