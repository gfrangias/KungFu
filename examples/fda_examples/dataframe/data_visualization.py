import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from data_analysis import data_analysis
from matplotlib.cm import get_cmap

def time_over_clients():
    da = data_analysis()

    da.get_last_epoch()

    da.group_repeated_expers(['clients','exper_type'])

    result_df = da.grouped_list[2]['time'].agg(['min', 'mean', 'max']).reset_index()

    print(result_df)

    result_grouped = result_df.groupby('exper_type')

    # Create the plot
    plt.figure()
    
    for exper_type, group in result_grouped:
        
        print(exper_type)
        print(group)

        x = group['clients']
        y_min = group['min']
        y_mean = group['mean']
        y_max = group['max']

        # Plot the mean line
        plt.plot(x, y_mean, label=exper_type)

        # Add the shaded region to represent the min-max range
        plt.fill_between(x, y_min, y_max, alpha=0.2)

    # Add labels and title
    plt.xlabel('Number of Clients')
    plt.ylabel('Duration')
    plt.title('Experiment Duration per Number of Clients')
    plt.legend()

    plt.savefig('duration_over_clients.svg')

    # Show the plot
    plt.show()

def syncs_over_clients():
    da = data_analysis()

    da.get_last_epoch()

    da.group_repeated_expers(['clients','exper_type'])

    result_df = da.grouped_list[2]['syncs'].agg(['min', 'mean', 'max']).reset_index()

    print(result_df)

    result_grouped = result_df.groupby('exper_type')

    # Create the plot
    plt.figure()
    
    for exper_type, group in result_grouped:
        
        print(exper_type)
        print(group)

        x = group['clients']
        y_min = group['min']
        y_mean = group['mean']
        y_max = group['max']

        # Plot the mean line
        plt.plot(x, y_mean, label=exper_type)

        # Add the shaded region to represent the min-max range
        plt.fill_between(x, y_min, y_max, alpha=0.2)

    # Add labels and title
    plt.xlabel('Number of Clients')
    plt.ylabel('Synchronizations')
    plt.title('Total Synchronizations per Number of Clients')
    plt.legend()

    plt.savefig('syncs_over_clients.svg')

    # Show the plot
    plt.show()

def syncs_over_clients_naive():
    da = data_analysis()

    da.get_last_epoch()

    da.group_repeated_expers(['clients','exper_type'])

    result_df = da.grouped_list[2]['syncs'].agg(['min', 'mean', 'max']).reset_index()

    print(result_df)

    result_grouped = result_df.groupby('exper_type')

    # Create the plot
    plt.figure()
    
    for exper_type, group in result_grouped:
        
        if exper_type == 'Naive FDA':
            print(exper_type)
            print(group)

            x = group['clients']
            y_min = group['min']
            y_mean = group['mean']
            y_max = group['max']

            # Plot the mean line
            plt.plot(x, y_mean, label=exper_type)

            # Add the shaded region to represent the min-max range
            plt.fill_between(x, y_min, y_max, alpha=0.2)

    # Add labels and title
    plt.xlabel('Number of Clients')
    plt.ylabel('Synchronizations')
    plt.title('Total Synchronizations per Number of Clients')
    plt.legend()

    plt.savefig('syncs_over_clients_naive.svg')

    # Show the plot
    plt.show()

def accuracy_over_epochs():
    da = data_analysis()

    da.group_repeated_expers(['clients','exper_type','epoch'], 2)

    result_df = da.grouped_list[0]['accuracy'].agg(['min', 'mean', 'max']).reset_index()

    print(result_df)

    result_grouped = result_df.groupby(['exper_type','clients'])

    # Create the plot
    plt.figure()

    # Get the colormaps
    cmap_red = get_cmap("Reds")
    cmap_blue = get_cmap("Blues")

    max_clients = result_df['clients'].max()
    
    for label, group in result_grouped:
        
        exper_type = label[0]
        clients = label[1]

        if exper_type == 'Naive FDA':
            color = cmap_blue((clients+4) / (2 * max_clients))
            color_fill = cmap_blue(1)
        elif exper_type == 'Synchronous SGD':
            color = cmap_red((clients+4) / (2 * max_clients))
            color_fill = cmap_red(1)
        
        x = group['epoch']
        y_min = group['min']
        y_mean = group['mean']
        y_max = group['max']

        # Plot the mean line
        plt.plot(x, y_mean, label=exper_type+' '+str(clients)+' clients', color=color)

        # Add the shaded region to represent the min-max range
        plt.fill_between(x, y_min, y_max, alpha=0.2, color=color)

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.savefig('accuracy_over_epochs.svg')

    # Show the plot
    plt.show()

def syncs_over_steps():
    da = data_analysis()

    da.group_repeated_expers(['clients','exper_type','step'], 1)

    result_df = da.grouped_list[0]['syncs'].agg(['min', 'mean', 'max']).reset_index()

    print(result_df)

    result_grouped = result_df.groupby(['exper_type','clients'])

    # Create the plot
    plt.figure()

    # Get the colormaps
    cmap_red = get_cmap("Reds")
    cmap_blue = get_cmap("Blues")

    max_clients = result_df['clients'].max()
    
    for label, group in result_grouped:
        
        exper_type = label[0]
        clients = label[1]

        if exper_type == 'Naive FDA':
            color = cmap_blue((clients+4) / (2 * max_clients))
            color_fill = cmap_blue(1)
        
            x = group['step']
            y_min = group['min']
            y_mean = group['mean']
            y_max = group['max']

            # Plot the mean line
            plt.plot(x, y_mean, label=exper_type+' '+str(clients)+' clients', color=color)

            # Add the shaded region to represent the min-max range
            plt.fill_between(x, y_min, y_max, alpha=0.2, color=color)

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Synchronizations')
    plt.title('Synchronizations over Steps')
    plt.legend()

    plt.savefig('syncs_over_steps.svg')

    # Show the plot
    plt.show()

def syncs_over_steps_rate():
    da = data_analysis()

    da.group_repeated_expers(['clients','exper_type','step'], 1)

    result_df = da.grouped_list[0]['syncs'].agg(['min', 'mean', 'max']).reset_index()

    # Divide min, mean, and max values by 'step' column value
    result_df['min'] = result_df['min'] / result_df['step']
    result_df['mean'] = result_df['mean'] / result_df['step']
    result_df['max'] = result_df['max'] / result_df['step']

    print(result_df)

    result_grouped = result_df.groupby(['exper_type','clients'])

    # Create the plot
    plt.figure()

    # Get the colormaps
    cmap_red = get_cmap("Reds")
    cmap_blue = get_cmap("Blues")

    max_clients = result_df['clients'].max()
    
    for label, group in result_grouped:
        
        exper_type = label[0]
        clients = label[1]

        if exper_type == 'Naive FDA':
            color = cmap_blue((clients+4) / (2 * max_clients))
            color_fill = cmap_blue(1)
        
            x = group['step']
            y_min = group['min']
            y_mean = group['mean']
            y_max = group['max']

            # Plot the mean line
            plt.plot(x, y_mean, label=exper_type+' '+str(clients)+' clients', color=color)

            # Add the shaded region to represent the min-max range
            plt.fill_between(x, y_min, y_max, alpha=0.2, color=color)

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Synchronizations')
    plt.title('Synchronizations over Steps Rate')
    plt.legend()

    plt.savefig('syncs_over_steps_rate.svg')

    # Show the plot
    plt.show()

def accuracy_over_time():
    da = data_analysis()

    da.group_repeated_expers(['clients', 'exper_type', 'time', 'epoch'], 2)

    result_df = da.grouped_list[0]['accuracy'].agg(['min', 'mean', 'max']).reset_index()

    result_df = result_df[result_df['epoch'] <= 20]

    print(result_df)
    
    # Create a list of distinct 'clients' values
    distinct_clients = result_df['clients'].unique()

    # Iterate through each distinct 'client' value and create a figure for it
    for client in distinct_clients:
        # Filter the DataFrame for rows where 'clients' column is equal to the current 'client'
        filtered_df = result_df[result_df['clients'] == client]

        # Create a new figure
        plt.figure()

        # Iterate through the subgroups of the filtered DataFrame to plot lines
        for key, group in filtered_df.groupby('exper_type'):
            if key == 'Synchronous SGD': 
                key='Synchronous'
                color = "#0072BD"
            else:
                color = "#D95319"

            plt.plot(group['time']/60, group['mean'], label=key, color=color)    

            # Get the last time value for this group
            last_time_value = group['time'].iloc[-1] / 60  # Convert to minutes
            last_accuracy_value = group['mean'].iloc[-1]
            
            # Annotate the last time value on the plot
            plt.text(last_time_value, last_accuracy_value, f"{last_time_value:.2f} mins", color=color) 

        nodes = str(int(int(client)/2))
        # Add title, labels, and legend
        if nodes == '1': plt.title(f"Accuracy over time for {nodes} node")
        else: plt.title(f"Accuracy over time for {nodes} nodes")
        plt.xlabel("Training Time (mins)")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig(f"accuracy_over_time_clients_{client}.svg")

        # Show the plot
        plt.show()


def syncs_over_epochs():
    da = data_analysis()

    da.group_repeated_expers(['clients', 'exper_type', 'syncs', 'epoch', 'steps'], 2)

    result_df = da.grouped_list[0]['syncs'].agg(['min', 'mean', 'max']).reset_index()

    result_df = result_df[result_df['epoch'] <= 20]

    print(result_df)
    
    # Create a list of distinct 'clients' values
    distinct_clients = result_df['clients'].unique()

    # Iterate through each distinct 'client' value and create a figure for it
    for client in distinct_clients:
        # Filter the DataFrame for rows where 'clients' column is equal to the current 'client'
        filtered_df = result_df[result_df['clients'] == client]
        filtered_df['syncs_each'] = filtered_df['mean'].diff()

        filtered_df['syncs_each'].iloc[0] = filtered_df['mean'].iloc[0]

        # Create a new figure
        plt.figure()

        # Iterate through the subgroups of the filtered DataFrame to plot lines
        for key, group in filtered_df.groupby('exper_type'):
            if key == 'Synchronous SGD': 
                continue
            else:
                color = "#0072BD"

            plt.plot(group['epoch']/60, group['syncs_each'], label=key, color=color)    

            last_syncs_value = str(int(group['mean'].iloc[-1]))
            last_steps_value = str(int(group['steps'].iloc[-1]))

            plt.text(0.5, 0.55, f"FDA total syncs {last_syncs_value}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color="#D95319" )
            plt.text(0.5, 0.5, f"Synchronous total syncs {last_steps_value}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color="#0072BD" )

        nodes = str(int(int(client)/2))
        # Add title, labels, and legend
        if nodes == '1': plt.title(f"Synchronizations over epochs for {nodes} node")
        else: plt.title(f"Synchronizations over epochs for {nodes} nodes")        
        plt.xlabel("Epochs")
        plt.ylabel("Synchronizations")
        plt.legend()

        plt.savefig(f"syncs_over_epochs_clients_{client}.svg")

        # Show the plot
        plt.show()


def combined_plot_to_single_svg(list):
    
    da_accuracy = data_analysis()
    da_syncs = data_analysis()
    
    da_accuracy.group_repeated_expers(['clients', 'exper_type', 'time', 'epoch'], 2)
    da_syncs.group_repeated_expers(['clients', 'exper_type', 'syncs', 'epoch', 'steps'], 2)
    
    result_df_accuracy = da_accuracy.grouped_list[0]['accuracy'].agg(['min', 'mean', 'max']).reset_index()
    result_df_syncs = da_syncs.grouped_list[0]['syncs'].agg(['min', 'mean', 'max']).reset_index()
    
    result_df_accuracy = result_df_accuracy[result_df_accuracy['epoch'] <= 20]
    result_df_syncs = result_df_syncs[result_df_syncs['epoch'] <= 20]
    
    distinct_clients = result_df_accuracy['clients'].unique()
    distinct_clients = [client for client in distinct_clients if client in list]
    num_clients = len(distinct_clients)

    fig, axes = plt.subplots(num_clients, 2, figsize=(15, 5*num_clients))

    for idx, client in enumerate(distinct_clients):
        # Accuracy subplot
        print(client)
        ax1 = axes[idx, 0]
        filtered_df_accuracy = result_df_accuracy[result_df_accuracy['clients'] == client]
        
        for key, group in filtered_df_accuracy.groupby('exper_type'):
            if key == 'Synchronous SGD': 
                key = 'Synchronous'
                color = "#0072BD"
            else:
                color = "#D95319"
            
            ax1.plot(group['time']/60, group['mean'], label=key, color=color)
            nodes = str(int(int(client)/2))

            # Get the last time value for this group
            last_time_value = group['time'].iloc[-1] / 60  # Convert to minutes
            last_accuracy_value = group['mean'].iloc[-1]
            
            # Annotate the last time value on the plot
            ax1.text(last_time_value, last_accuracy_value, f"{last_time_value:.2f} mins", color=color)
            if nodes == '1':
                ax1.text(0.5, 0.5, f"{nodes} node, 2 GPUs per node", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color="#4DBEEE" )
            else:
                ax1.text(0.5, 0.5, f"{nodes} nodes, 2 GPUs per node", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color="#4DBEEE" )


        ax1.set_xlabel("Training Time (mins)")
        ax1.set_ylabel("Accuracy")
        ax1.legend(loc='lower right')
        
        # Syncs subplot
        ax2 = axes[idx, 1]
        filtered_df_syncs = result_df_syncs[result_df_syncs['clients'] == client]
        filtered_df_syncs['syncs_each'] = filtered_df_syncs['mean'].diff()
        filtered_df_syncs['syncs_each'].iloc[0] = filtered_df_syncs['mean'].iloc[0]
        
        for key, group in filtered_df_syncs.groupby('exper_type'):
            if key == 'Synchronous SGD': 
                continue
            else:
                color = "#0072BD"

            ax2.plot(group['epoch'], group['syncs_each'], label=key, color=color) 
            last_syncs_value = str(int(group['mean'].iloc[-1]))
            last_steps_value = str(int(group['steps'].iloc[-1]))

            ax2.text(0.5, 0.55, f"FDA total syncs {last_syncs_value}", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, color="#D95319" )
            ax2.text(0.5, 0.5, f"Synchronous total syncs {last_steps_value}", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, color="#0072BD" )

        
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Synchronizations")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(f"combined_plot_clients_{list[0]}_{list[1]}.svg")
    plt.show()
