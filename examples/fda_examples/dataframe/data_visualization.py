import matplotlib.pyplot as plt
import pandas
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

