import argparse
import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import group_all_reduce
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
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss = tf.losses.SparseCategoricalCrossentropy()

# KungFu: adjust learning rate based on number of GPUs.
# opt = tf.keras.optimizers.SGD(0.001 * current_cluster_size())
opt = tf.compat.v1.train.AdamOptimizer(0.001 * current_cluster_size())

steps_since_sync = tf.Variable(0, dtype=tf.int32)
total_syncs = tf.Variable(0, dtype=tf.int32)
syncs_hist = []
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
    models_diff = tensor_lists_subtraction(local_model, last_sync_model)   # Convert the list of tensors into a single tensor
    models_diff_tensor = tensor_list_to_vector(models_diff)                # Subtract the two models
    norm_2 =  tf.norm(models_diff_tensor, 2)                               # Compute the 2-norm of the subtraction
    return tensor_to_tensor_list(norm_2)

def compute_xi(second_last_sync_model, last_sync_model):
    sync_models_diff = tensor_lists_subtraction(last_sync_model, second_last_sync_model)
    sync_models_diff_tensor = tensor_list_to_vector(sync_models_diff)
    sync_models_diff_tensor = tf.abs(sync_models_diff_tensor)
    xi = tf.divide(sync_models_diff_tensor, tf.norm(sync_models_diff_tensor))
    return tensor_to_tensor_list(xi)

def compute_xi_2(second_last_sync_model, last_sync_model):
    sync_models_diff = tensor_lists_subtraction(last_sync_model, second_last_sync_model)
    sync_models_diff_tensor = tensor_list_to_vector(sync_models_diff)
    sync_models_diff_tensor = tf.abs(sync_models_diff_tensor)
    xi = tf.divide(sync_models_diff_tensor, tf.norm(sync_models_diff_tensor))
    return tensor_to_tensor_list(xi)

@tf.function
def compute_xi_dot_diff(last_sync_model, local_model, xi):
    models_diff = tensor_lists_subtraction(local_model, last_sync_model)
    models_diff_tensor = tensor_list_to_vector(models_diff)
    if tf.is_tensor(xi[0]):
        xi_tensor = tensor_list_to_vector(xi)
    else:
        random_tensor = tf.random.normal(shape=(len(models_diff_tensor),), dtype=tf.float32)
        xi_tensor = tf.divide(random_tensor, tf.norm(random_tensor))

    xi_dot_diff = tf.tensordot(xi_tensor, models_diff_tensor, axes=1)
    return tensor_to_tensor_list(xi_dot_diff)

@tf.function
def rtc_check(divergence):
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

    summed_divergences = group_all_reduce(local_divergence)
    np = tf.cast(current_cluster_size(), tf.float32)
    averaged_divergence = map_maybe(lambda d: d / np, summed_divergences)

    if rtc_check(averaged_divergence) or first_batch:
        if current_rank() == 0:
            tf.print("Steps since last sync: ", steps_since_sync)
            tf.print("Average divergence: ", averaged_divergence)
        reset_counter()
        total_syncs.assign_add(1)
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
    local_xi_dot_diff = compute_xi_dot_diff(last_sync_model, mnist_model.trainable_variables, xi)

    summed_divergences = group_all_reduce(local_divergence)
    summed_xi_dot_diff = group_all_reduce(local_xi_dot_diff)
    np = tf.cast(current_cluster_size(), tf.float32)
    averaged_divergence = map_maybe(lambda d: d / np, summed_divergences)
    averaged_xi_dot_diff = map_maybe(lambda d: d / np, summed_xi_dot_diff)
    #tf.print(averaged_xi_dot_diff)
    rtc_expr = averaged_divergence - tf.square(averaged_xi_dot_diff)

    #tf.print(rtc_expr)
    if rtc_check(rtc_expr) or first_batch:
        if current_rank() == 0:
            tf.print("Steps since last sync: ", steps_since_sync)
            tf.print("Average divergence: ", rtc_expr)
        reset_counter()
        total_syncs.assign_add(1)
        summed_models = group_all_reduce(mnist_model.trainable_variables)
        # Cast the number of workers to tf.float32
        np = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / np, summed_models)
        for i, variable in enumerate(mnist_model.trainable_variables):
            variable.assign(averaged_models[i])
        # Compute the difference between the latest updated model and the second latest updated model
        xi = compute_xi(updated_last_sync_model, mnist_model.trainable_variables)
        updated_last_sync_model = mnist_model.trainable_variables

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_batch:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(mnist_model.variables)
        broadcast_variables(opt.variables())

    return loss_value, updated_last_sync_model, xi

_ = mnist_model(tf.random.normal((1, 28, 28, 1), dtype=tf.float32))
last_sync_model = [0.0] * len(mnist_model.trainable_variables)
xi = [0.0]
# KungFu: adjust number of steps based on number of GPUs.
for batch, (images, labels) in enumerate(
        dataset.take(5000 // current_cluster_size())):
    
    #print("Step #"+str(batch))
    if args.fda == 'naive':
        loss_value, last_sync_model = training_step_naive(images, labels, batch == 0, last_sync_model)
    elif args.fda == 'linear':
        loss_value, last_sync_model, xi = training_step_linear(images, labels, batch == 0, last_sync_model, xi)
    elif args.fda == 'sketch':
        print("Sketch not implemented yet!")
        exit()
    else:
        print("FDA method \""+args.fda+"\" isn't available!")
        exit()

    syncs_hist.append(total_syncs)
    if current_rank() == 0:
        print('%d,%d' % (batch, syncs_hist[-1]))
    #if batch % 10 == 0 and current_rank() == 0:
        #print('Step #%d\tLoss: %.6f and %d synchronizations occured.' % (batch, loss_value, syncs_hist[-1]))