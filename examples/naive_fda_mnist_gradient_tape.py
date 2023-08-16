import argparse

import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import (defuse, fuse, group_all_reduce,
                                   group_nccl_all_reduce, monitored_all_reduce,
                                   peer_info, set_tree, broadcast, subset_all_reduce,
                                   current_rank)
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer,
                                          MySynchronousSGDOptimizer)

parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--kf-optimizer',
                    type=str,
                    default='my-sync-sgd',
                    help='available options: sync-sgd, async-sgd, sma, my-sync-sgd')
parser.add_argument('--fda',
                    type=str,
                    default='naive',
                    help='available options: naive, linear, sketch')
parser.add_argument('--threshold',
                    type=str,
                    default='0.8')
args = parser.parse_args()

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % current_rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0,
             tf.float32), tf.cast(mnist_labels, tf.int64)))

dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the input image
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit) and softmax activation
])

loss = tf.losses.SparseCategoricalCrossentropy()

# KungFu: adjust learning rate based on number of GPUs.
# opt = tf.keras.optimizers.SGD(0.001 * current_cluster_size())
opt = tf.compat.v1.train.AdamOptimizer(0.001 * current_cluster_size())

steps_since_sync = tf.Variable(0, dtype=tf.int32)
threshold = float(args.threshold)
threshold = tf.constant(threshold)

# KungFu: wrap tf.compat.v1.train.Optimizer.
if args.kf_optimizer == 'sync-sgd':
    opt = SynchronousSGDOptimizer(opt)
elif args.kf_optimizer == 'async-sgd':
    opt = PairAveragingOptimizer(opt)
elif args.kf_optimizer == 'sma':
    opt = SynchronousAveragingOptimizer(opt)
elif args.kf_optimizer == 'my-sync-sgd':
    opt = MySynchronousSGDOptimizer(opt)
else:
    raise RuntimeError('Unknown KungFu optimizer')

@tf.function
def tensor_lists_subtraction(l1,l2):
    return [tf.subtract(t1, t2) for t1, t2 in zip(l1, l2)]

@tf.function
def tensor_list_to_vector(tensor_list):
    return tf.concat([tf.reshape(var, [-1]) for var in tensor_list], axis=0)

@tf.function
def tensor_to_tensor_list(tensor):
    tensor_list = []
    tensor_list.append(tensor)
    return tensor_list

@tf.function
def compute_divergence_2_norm(last_sync_model, local_model):
    subtraction_list = tensor_lists_subtraction(local_model, last_sync_model)   # Convert the list of tensors into a single tensor
    subtraction = tensor_list_to_vector(subtraction_list)                       # Subtract the two models
    norm_2 =  tf.norm(subtraction, 2)                                           # Compute the 2-norm of the subtraction
    return tensor_to_tensor_list(norm_2)

@tf.function
def compute_xi(second_last_sync_model, last_sync_model):
    subtraction_list = tensor_lists_subtraction()
    xi = [t / tf.norm(tf.concat(subtraction_list, axis=0)) for t in subtraction_list]
    return xi

@tf.function
def should_synchronize_models_naive(divergence):
    global threshold
    if tf.math.greater(tf.cast(divergence, tf.float32),tf.cast(threshold, tf.float32)):
        return True
    else:
        return False
    
@tf.function
def increment_counter():
    steps_since_sync.assign_add(1)
    
@tf.function
def reset_counter():
    steps_since_sync.assign(0)

@tf.function
def training_step_naive(images, labels, first_batch, last_sync_model):
    increment_counter()
    # Perform TensorFlow GradientTape
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Find the local gradients 
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)

    # Apply the new gradients locally
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    updated_last_sync_model = last_sync_model

    local_divergence = compute_divergence_2_norm(last_sync_model, mnist_model.trainable_variables)
    print(local_divergence)
    summed_divergences = group_all_reduce(local_divergence)
    np = tf.cast(current_cluster_size(), tf.float32)
    averaged_divergence = map_maybe(lambda d: d / np, summed_divergences)

    if should_synchronize_models_naive(averaged_divergence) or first_batch:
        tf.print("Steps since last sync: ", steps_since_sync)
        tf.print("Average divergence: ", averaged_divergence)
        reset_counter()
        summed_models = group_all_reduce(mnist_model.trainable_variables)
        # Cast the number of workers to tf.float32
        np = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / np, summed_models)
        for i, variable in enumerate(mnist_model.trainable_variables):
            variable.assign(averaged_models[i])
        updated_last_sync_model = mnist_model.trainable_variables
        
    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_batch:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(mnist_model.variables)
        broadcast_variables(opt.variables())

    return loss_value, updated_last_sync_model

@tf.function
def training_step_linear(images, labels, first_batch, last_sync_model, xi):
    increment_counter()
    # Perform TensorFlow GradientTape
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Find the local gradients 
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)

    # Apply the new gradients locally
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    updated_last_sync_model = last_sync_model

    local_divergence = compute_divergence_2_norm(last_sync_model, mnist_model.trainable_variables)
    print(local_divergence)
    summed_divergences = group_all_reduce(local_divergence)
    np = tf.cast(current_cluster_size(), tf.float32)
    averaged_divergence = map_maybe(lambda d: d / np, summed_divergences)

    if should_synchronize_models_naive(averaged_divergence) or first_batch:
        tf.print("Steps since last sync: ", steps_since_sync)
        tf.print("Average divergence: ", averaged_divergence)
        reset_counter()
        summed_models = group_all_reduce(mnist_model.trainable_variables)
        # Cast the number of workers to tf.float32
        np = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / np, summed_models)
        for i, variable in enumerate(mnist_model.trainable_variables):
            variable.assign(averaged_models[i])
        xi = compute_xi(updated_last_sync_model, mnist_model.trainable_variables)   # Compute the unit vector needed for Linear FDA
        updated_last_sync_model = mnist_model.trainable_variables
        
    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_batch:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(mnist_model.variables)
        broadcast_variables(opt.variables())

    return loss_value, updated_last_sync_model, xi

last_sync_model = [0.0] * len(mnist_model.trainable_variables)

# KungFu: adjust number of steps based on number of GPUs.
for batch, (images, labels) in enumerate(
        dataset.take(10000 // current_cluster_size())):
    
    #print("Step #"+str(batch))
    if args.fda == 'naive':
        loss_value, last_sync_model = training_step_naive(images, labels, batch == 0, last_sync_model)
    elif args.fda == 'linear':
        loss_value, last_sync_model = training_step_linear(images, labels, batch == 0, last_sync_model)
    elif args.fda == 'sketch':
        print("Sketch not implemented yet!")
        exit()
    else:
        print("FDA method \""+args.fda+"\" isn't available!")
        exit()

    if batch % 10 == 0 and current_rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, loss_value))
