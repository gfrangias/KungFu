def map_epochs(exper_type, model_type, batch_size, threshold=None):
    epochs = 0

    if exper_type == "Synchronous SGD":
        if model_type == "lenet5":
            match batch_size:
                case 32: epochs = 250
                case 64: epochs = 400
                case 128: epochs = 750
                case 256: epochs = 1500
    elif exper_type == "Naive FDA":
        epochs = 0

    
    return str(epochs)

def get_sublist(lst, i):
    n = len(lst)
    base_size = n // 4
    remainder = n % 4

    start = 0
    for j in range(4):
        end = start + base_size + (1 if j < remainder else 0)
        if j == i:
            return lst[start:end]
        start = end