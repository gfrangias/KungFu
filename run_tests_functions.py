def map_epochs(exper_type, model_type, batch_size, threshold=None):
    epochs = None

    if exper_type == "Synchronous SGD":
        if model_type == "lenet5":
            if batch_size == "32": epochs = 250
            if batch_size == "64": epochs = 400
            if batch_size == "128": epochs = 750
            if batch_size == "256": epochs = 1500
        elif model_type == "adv_cnn":
            if batch_size == "32": epochs = 300
            if batch_size == "64": epochs = 500
            if batch_size == "128": epochs = 1000
            if batch_size == "256": epochs = 2000

    elif exper_type == "Naive FDA":
        if model_type == "lenet5":
            if threshold == "0.5":
                if batch_size == "32": epochs = 350
                if batch_size == "64": epochs = 600
                if batch_size == "128": epochs = 1000
                if batch_size == "256": epochs = 2000
            elif threshold == "1.0":
                if batch_size == "32": epochs = 400
                if batch_size == "64": epochs = 700
                if batch_size == "128": epochs = 1500
                if batch_size == "256": epochs = 3000

    return str(epochs)

def distribute_clients(clients, nodes):
    # Calculate the base number of clients per node
    base_clients_per_node = clients // nodes

    # Calculate the number of nodes that will get one extra client
    nodes_with_extra_client = clients % nodes

    # Create a list to store the number of clients in each node
    clients_per_node = [base_clients_per_node] * nodes

    # Distribute the extra clients to the first few nodes
    for i in range(nodes_with_extra_client):
        clients_per_node[i] += 1

    return clients_per_node

