def distribute_clients(clients, nodes):
    # Calculate the base number of clients per node
    base_clients_distr = clients // nodes

    # Calculate the number of nodes that will get one extra client
    nodes_with_extra_client = clients % nodes

    # Create a list to store the number of clients in each node
    clients_distr = [base_clients_distr] * nodes

    # Distribute the extra clients to the first few nodes
    for i in range(nodes_with_extra_client):
        clients_distr[i] += 1

    return clients_distr

