import pandas as pd
import glob

class data_analysis:

    def __init__(self, directory):

        # Directory where all CSV files are located
        self.directory = "../csv_files/"+directory+"/"
        
        self.df = {}
        self.grouped_df = {}
        
        # Create experiments info dataframe
        info_file = self.directory + "info.csv"
        self.df['info'] = pd.read_csv(info_file)

        epoch_files = self.directory + "epoch_step_info/*epoch.csv"
        step_files = self.directory + "epoch_step_info/*step.csv"

        # Combine all files for steps and epochs in two dataframes
        self.df['epoch'], self.df['step'] = self.concat_epoch_step_files(epoch_files, step_files)

        # Add for each step and epoch dataframe row all the experiment info
        self.epoch_step_with_info()

    # Combine all files for steps and epochs in two dataframes
    def concat_epoch_step_files(self, epoch_files, step_files):
        epoch_files_list = glob.glob(epoch_files)
        step_files_list = glob.glob(step_files)

        epoch_dfs, step_dfs = [], []

        for epoch_file, step_file in zip(epoch_files_list, step_files_list):
            
            epoch_file_df = pd.read_csv(epoch_file)
            step_file_df = pd.read_csv(step_file)
            
            epoch_dfs.append(epoch_file_df)
            step_dfs.append(step_file_df)    

        epoch_df = pd.concat(epoch_dfs, ignore_index=True)
        step_df = pd.concat(step_dfs, ignore_index=True)

        return epoch_df, step_df
    
    # Add for each step and epoch dataframe row all the experiment info
    def epoch_step_with_info(self):

        self.df['epoch'] = pd.merge(self.df['info'], self.df['epoch'], on='exper_id')
        self.df['step'] = pd.merge(self.df['info'], self.df['step'], on='exper_id')

    # Select from dataframes the rows that have specific values in columns
    def select_where(self, selections):
        columns = list(selections.keys())
        values = list(selections.values())
        threshold_in_columns = False

        # Check if 'threshold' is in the selections
        if 'threshold' in columns:
            threshold_in_columns = True
            threshold_index = columns.index('threshold')
            threshold_value = values.pop(threshold_index)
            columns.remove('threshold')
        
        for key, df in self.df.items():
            conditions = [df[column] == value for column, value in zip(columns, values)]
            
            if threshold_in_columns:
                threshold_condition = (df['threshold'] == threshold_value) | (df['threshold'].isna())
                conditions.append(threshold_condition)

            final_condition = conditions[0]

            for condition in conditions[1:]:
                final_condition &= condition
            self.df[key] = df[final_condition]

    def group_repeated_expers(self, attributes, key, aggr, time = False):
        # Group by multiple columns
        grouped_df = self.df[key].groupby(attributes, dropna=False)
        if time:
            resulting_df = grouped_df.agg({aggr: ['min', 'mean', 'max'], 'time': 'mean'}).reset_index()
            resulting_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in resulting_df.columns]
        else:
            resulting_df = grouped_df['time'].agg(['min', 'mean', 'max']).reset_index()
        return resulting_df